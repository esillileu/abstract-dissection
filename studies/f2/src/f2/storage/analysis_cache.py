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
    from repro_mlflow.artifact_cache import MlflowArtifactCache

    from .publication import PublicationService

    publication = PublicationService(tracking_uri)
    client = publication.client
    cache_root = getattr(paths, "mlflow_cache_root", paths.cache_root / "mlflow")
    cache = MlflowArtifactCache(client, tracking_uri, root=cache_root)

    root = (
        paths.analysis_cache("f2", suite)
        if hasattr(paths, "analysis_cache")
        else paths.cache_root / "f2" / suite / "results"
    )
    root.mkdir(parents=True, exist_ok=True)
    selected_finished = publication.get_finished_runs_for_experiment(f"f2.{suite}")

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
                if remote_paths is None or "evaluation" in remote_paths:
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
