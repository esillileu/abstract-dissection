"""Download MLflow results and read the resulting disposable local snapshot."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any


def download_results(tracking_uri: str, suite: str, paths) -> Path:
    """Refresh run metadata and download analysis inputs before invoking an analyzer.

    Selects the latest FINISHED run for each experimental condition.
    Artifacts are downloaded into workspace/analysis/cache/mlflow/.
    Metadata snapshot is recorded in workspace/analysis/cache/results/f2/<suite>/.
    """
    from mlflow import MlflowClient

    from repro_mlflow.artifact_cache import MlflowArtifactCache

    client = MlflowClient(tracking_uri=tracking_uri)
    cache_root = getattr(paths, "mlflow_cache_root", paths.cache_root / "mlflow")
    cache = MlflowArtifactCache(client, tracking_uri, root=cache_root)

    root = (
        paths.analysis_cache("f2", suite)
        if hasattr(paths, "analysis_cache")
        else paths.cache_root / "f2" / suite / "results"
    )
    root.mkdir(parents=True, exist_ok=True)
    experiment = client.get_experiment_by_name(f"f2.{suite}")

    all_runs: list[Any] = []
    token = None
    while experiment is not None:
        page = client.search_runs(
            [experiment.experiment_id],
            order_by=["attributes.start_time DESC"],
            max_results=1000,
            page_token=token,
        )
        all_runs.extend(page)
        token = getattr(page, "token", None)
        if not token:
            break

    def _run_training_settings(r: Any) -> tuple[Any, ...]:
        t = r.data.tags
        p = r.data.params
        variant = t.get("implementation.variant", "")
        return (
            t.get("config_digest") or t.get("f2.config_digest", ""),
            p.get("model_kind") or t.get("training.model_kind", ""),
            p.get("corpus_id")
            or t.get("corpus.id")
            or t.get("f2.source_corpus_sha256", ""),
            t.get("is_phrase") or str("phrase" in variant),
            p.get("objective_kind") or t.get("training.objective", ""),
            p.get("embedding_dimension") or p.get("dim") or t.get("training.dim", ""),
            p.get("window_radius")
            or p.get("window_size")
            or t.get("training.window", ""),
            p.get("subsampling_threshold") or t.get("training.subsample", ""),
            p.get("negative_sample_count")
            or p.get("negative_samples")
            or t.get("training.negative_samples", ""),
            p.get("epochs") or t.get("training.epochs", ""),
            t.get("phrase.delta", ""),
            t.get("phrase.threshold", ""),
        )

    # 1. Group finished runs by target experimental condition (spec_id and variant)
    runs_by_target: dict[tuple[str, str], list[Any]] = {}
    for run in all_runs:
        if run.info.status == "FINISHED":
            tags = run.data.tags
            spec_id = tags.get("experiment_spec.id", "")
            variant = tags.get("implementation.variant", "")
            target_key = (spec_id, variant)
            runs_by_target.setdefault(target_key, []).append(run)

    # 2. For each target, determine actual latest training settings and select latest finished run per seed
    selected_finished: list[Any] = []
    for target_key in sorted(runs_by_target):
        target_runs = runs_by_target[target_key]
        if not target_runs:
            continue
        # Latest finished run for this experimental target defines actual training conditions
        latest_target_run = max(
            target_runs,
            key=lambda r: (
                r.info.end_time or 0,
                r.info.start_time or 0,
                r.info.run_id,
            ),
        )
        target_settings = _run_training_settings(latest_target_run)

        # Do not mix different training settings even for same variant and seed
        matching_runs = [
            r for r in target_runs if _run_training_settings(r) == target_settings
        ]

        # Group by seed and pick the latest finished run
        seeds_map: dict[str, list[Any]] = {}
        for r in matching_runs:
            seed = r.data.tags.get("seed") or r.data.params.get("seed", "")
            seeds_map.setdefault(seed, []).append(r)

        for seed in sorted(seeds_map):
            best = max(
                seeds_map[seed],
                key=lambda r: (
                    r.info.end_time or 0,
                    r.info.start_time or 0,
                    r.info.run_id,
                ),
            )
            selected_finished.append(best)

    records: list[dict[str, Any]] = []
    for run in selected_finished:
        record = {
            "run_id": run.info.run_id,
            "tags": dict(run.data.tags),
            "status": run.info.status,
            "end_time": run.info.end_time,
            "start_time": run.info.start_time,
        }
        try:
            result_path = cache.get(run.info.run_id, "result.json")
            record["artifact_root"] = str(result_path.parent)
            result = json.loads(result_path.read_text())
            cache.get(run.info.run_id, result["artifacts"]["lookup"])
            try:
                cache.get(run.info.run_id, "evaluation")
            except Exception:
                pass
            for name in (
                "metrics/observations.csv",
                "environment/system.json",
                "data/dataset_manifest.json",
                "phrase_lineage.json",
                "large_scale_lineage.json",
            ):
                try:
                    cache.get(run.info.run_id, name)
                except Exception:
                    pass
        except Exception as exc:
            record["error"] = f"artifact retrieval failed: {type(exc).__name__}"
        records.append(record)

    temporary = root / ".runs.json.tmp"
    temporary.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")
    temporary.replace(root / "runs.json")
    return root


def read_records(root: Path) -> list[dict]:
    path = Path(root) / "runs.json"
    return json.loads(path.read_text()) if path.is_file() else []


class LocalResults:
    """Local run metadata and artifact paths; this reader cannot perform network I/O."""

    def __init__(self, root: Path):
        self.records = read_records(root)
        self.roots = {
            r["run_id"]: Path(r["artifact_root"])
            for r in self.records
            if r.get("artifact_root")
        }

    def runs(self):
        return [
            SimpleNamespace(
                info=SimpleNamespace(run_id=r["run_id"], status=r["status"]),
                data=SimpleNamespace(tags=r["tags"]),
            )
            for r in self.records
            if r["status"] == "FINISHED"
        ]

    def get(self, run_id: str, artifact_path: str) -> Path:
        root = self.roots[run_id].resolve()
        target = (root / artifact_path).resolve()
        if not target.is_relative_to(root):
            raise ValueError("artifact path escapes cache run directory")
        if not target.exists():
            raise FileNotFoundError(target)
        return target


def unavailable_report(output: Path, reason: str) -> Path:
    """Record missing analysis inputs without creating experimental run state."""
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.md").write_text(f"# Analysis unavailable\n\n{reason}\n")
    (output / "unavailable.json").write_text(
        json.dumps({"status": "no_data", "reason": reason}, indent=2) + "\n"
    )
    return output


__all__ = ["LocalResults", "download_results", "read_records", "unavailable_report"]
