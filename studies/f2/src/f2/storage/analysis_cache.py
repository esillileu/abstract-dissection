"""Download MLflow results and read the resulting disposable local snapshot."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any


def download_results(
    tracking_uri: str,
    suite: str,
    paths,
    *,
    table: int | str | None = None,
    need_evaluation_benchmarks: bool | None = None,
) -> Path:
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

    # Group finished runs by Condition Identity:
    # (suite_id, experiment_id, condition_id, planned_run_slot_id, config_digest, seed)
    # Ref.md 3.3 / P0-2: run, check, and download_results must use the exact same Condition Identity.
    # We select the latest FINISHED run for each distinct experimental condition and seed.
    finished_by_slot_seed: dict[tuple[str, str, str], list[Any]] = {}
    for run in all_runs:
        if run.info.status != "FINISHED":
            continue
        tags = run.data.tags
        slot = tags.get("f2.planned_run_slot_id") or tags.get("planned_run_slot_id", "")
        digest = tags.get("f2.config_digest") or tags.get("config_digest", "")
        seed = str(
            tags.get("f2.seed") or tags.get("seed") or run.data.params.get("seed", "")
        )
        variant = (
            tags.get("implementation.variant")
            or tags.get("atomic_run_id")
            or tags.get("f2.atomic_run_id", "")
        )
        # If planned_run_slot_id and config_digest are present, use canonical identity key
        if slot and digest:
            key = (slot, digest, seed)
        else:
            # Fallback for runs without slot tags
            spec_id = tags.get("experiment_spec.id", "")
            key = (f"{spec_id}:{variant}", digest, seed)

        finished_by_slot_seed.setdefault(key, []).append(run)

    selected_finished: list[Any] = []
    for key in sorted(finished_by_slot_seed, key=str):
        candidate_runs = finished_by_slot_seed[key]
        best = max(
            candidate_runs,
            key=lambda r: (
                r.info.end_time or 0,
                r.info.start_time or 0,
                r.info.run_id,
            ),
        )
        selected_finished.append(best)

    benchmarks = paths.inputs("f2", "benchmarks") if hasattr(paths, "inputs") else None
    if need_evaluation_benchmarks is None:
        need_evaluation_benchmarks = True

    runs_to_process = list(selected_finished)

    from tqdm import tqdm

    records: list[dict[str, Any]] = []
    pbar = tqdm(runs_to_process, desc="Syncing MLflow artifacts", unit="run")
    for run in pbar:
        tags = dict(run.data.tags)
        variant = tags.get("implementation.variant", run.info.run_id[:8])
        seed = tags.get("seed", "")
        pbar.set_postfix_str(f"{variant} (s{seed})" if seed else variant)
        record = {
            "run_id": run.info.run_id,
            "tags": tags,
            "status": run.info.status,
            "end_time": run.info.end_time,
            "start_time": run.info.start_time,
        }
        try:
            result_path = cache.get(run.info.run_id, "result.json")
            record["artifact_root"] = str(result_path.parent)
            result = json.loads(result_path.read_text())
            cache.get(run.info.run_id, result["artifacts"]["lookup"])
            remote_paths: set[str] | None = None
            try:
                manifest_path = cache.get(run.info.run_id, "result_manifest.json")
                manifest = json.loads(manifest_path.read_text())
                remote_paths = {f["path"] for f in manifest.get("files", [])}
                for p in list(remote_paths):
                    if "/" in p:
                        remote_paths.add(p.split("/", 1)[0])
            except Exception:
                pass
            if need_evaluation_benchmarks:
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
                if remote_paths is not None and name not in remote_paths:
                    continue
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
