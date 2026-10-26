"""MLflow publication and discovery contracts for F2 Word2Vec runs."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

from mlflow import MlflowClient
from mlflow.entities import Metric, Run

from f2.run_identity import RunIdentity


def find_matching_finished_run(
    client: MlflowClient,
    experiment_name: str,
    config_digest: str,
    seed: int,
) -> Run | None:
    """Find a finished MLflow run matching identical semantic experiment conditions."""
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return None

    runs = client.search_runs(
        [experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        max_results=10000,
    )
    for run in runs:
        tags = run.data.tags
        digest = tags.get("config_digest") or tags.get("f2.config_digest")
        run_seed = tags.get("seed")
        if digest == config_digest and run_seed == str(seed):
            return run
    return None


def publish_run_to_mlflow(
    client: MlflowClient,
    experiment_name: str,
    config: dict[str, Any],
    run_root: Path,
    *,
    include_checkpoints: bool = False,
) -> str:
    """Publish completed local run artifacts and metrics to MLflow.

    Marks the run FINISHED only after upload completes successfully.
    If an upload error occurs, the run is terminated as FAILED and the error is raised.
    Local files are always preserved.
    """
    experiment = client.get_experiment_by_name(experiment_name)
    experiment_id = (
        client.create_experiment(experiment_name)
        if experiment is None
        else experiment.experiment_id
    )

    identity = config["identity"]
    suite = _suite(config)
    run_identity = RunIdentity(
        planned_run_slot_id=str(identity["planned_run_slot_id"]),
        plan_revision=_plan_revision(str(identity["execution_plan_id"])),
        config_digest=str(identity["config_digest"]),
    )
    training = (
        config.get("training", {}) if isinstance(config.get("training"), dict) else {}
    )
    corpus = config.get("corpus", {}) if isinstance(config.get("corpus"), dict) else {}
    model_kind = str(training.get("model_kind", ""))
    corpus_id = str(corpus.get("id", corpus.get("name", "")))
    is_phrase = bool(
        config.get("phrase", False) or "phrase" in str(config.get("atomic_run_id", ""))
    )
    objective_kind = str(training.get("objective_kind", ""))
    dimension = str(
        training.get("embedding_dimension")
        or training.get("hidden_size")
        or training.get("dim", "")
    )
    window_radius = str(
        training.get("window_radius") or training.get("window_size", "")
    )
    subsampling = str(training.get("subsampling_threshold", ""))
    negative_sample_count = str(
        training.get("negative_sample_count") or training.get("negative_samples", "")
    )
    epochs = str(training.get("epochs") or training.get("max_epochs", ""))

    tags = {
        **run_identity.tags(),
        "f2.source_corpus_sha256": str(config.get("corpus", {}).get("sha256", "")),
        "paper.id": "mikolov-2013-efficient-estimation"
        if suite == "w2v1"
        else "mikolov-2013-distributed-representations",
        "suite.name": suite,
        "experiment_spec.id": str(
            identity.get(
                "experiment_spec_id",
                "w2v1-table2-cbow"
                if suite == "w2v1"
                else "w2v2-phrase-skipgram-1b-objectives",
            )
        ),
        "implementation.variant": str(config.get("atomic_run_id", "")),
        "seed": str(identity["seed"]),
        "training.model_kind": model_kind,
        "corpus.id": corpus_id,
        "is_phrase": str(is_phrase),
        "training.objective": objective_kind,
        "training.dim": dimension,
        "training.window": window_radius,
        "training.subsample": subsampling,
        "training.negative_samples": negative_sample_count,
        "training.epochs": epochs,
    }

    run = client.create_run(
        experiment_id,
        start_time=int(time.time() * 1000),
        tags=tags,
        run_name=str(identity["planned_run_slot_id"]),
    )
    run_id = run.info.run_id

    # Log params using actual configuration keys and query aliases
    for p_key, p_val in {
        "model_kind": model_kind,
        "corpus_id": corpus_id,
        "objective_kind": objective_kind,
        "embedding_dimension": dimension,
        "window_radius": window_radius,
        "subsampling_threshold": subsampling,
        "negative_sample_count": negative_sample_count,
        "dim": dimension,
        "window_size": window_radius,
        "negative_samples": negative_sample_count,
        "epochs": epochs,
        "seed": str(identity["seed"]),
    }.items():
        if p_val:
            client.log_param(run_id, p_key, p_val)

    try:
        # Identify designated final model path from result.json (if present)
        result_file = run_root / "result.json"
        final_lookup_rel: Path | None = None
        if result_file.is_file():
            try:
                res_data = json.loads(result_file.read_text(encoding="utf-8"))
                lookup_val = res_data.get("artifacts", {}).get("lookup")
                if lookup_val:
                    final_lookup_rel = Path(lookup_val)
            except Exception:
                pass

        def should_upload(file_path: Path) -> bool:
            rel = file_path.relative_to(run_root)
            if "preparation" in rel.parts:
                return False
            # Intermediate pointers (latest.json, best.json, etc.) and hidden/tmp files remain local only
            if (
                file_path.name in {"latest.json", "best.json"}
                or file_path.name.startswith(".")
                or file_path.name.endswith(".tmp")
            ):
                return False
            # Always upload the designated final model
            if final_lookup_rel is not None and (
                rel == final_lookup_rel or final_lookup_rel in rel.parents
            ):
                return True
            # Intermediate checkpoints (generations or periodic/epoch checkpoints) remain local only
            if "generations" in rel.parts or any(
                part.startswith("latest-epoch-")
                or part.startswith("periodic-")
                or part.startswith("best-")
                or part.startswith("epoch-")
                for part in rel.parts
            ):
                return False
            if not include_checkpoints and "checkpoints" in rel.parts:
                return False
            return True

        # Write local result manifest matching exactly what is uploaded
        manifest_path = run_root / "result_manifest.json"
        manifest_files = []
        for path in sorted(item for item in run_root.rglob("*") if item.is_file()):
            if path == manifest_path or not should_upload(path):
                continue
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            manifest_files.append(
                {
                    "path": path.relative_to(run_root).as_posix(),
                    "size": path.stat().st_size,
                    "sha256": digest.hexdigest(),
                }
            )
        manifest_payload = {"files": manifest_files, "checkpoints": []}
        manifest_path.write_text(
            json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        # Upload artifacts matching result manifest
        for path in sorted(item for item in run_root.rglob("*") if item.is_file()):
            if not should_upload(path) and path != manifest_path:
                continue
            parent = path.parent.relative_to(run_root)
            client.log_artifact(
                run_id,
                str(path),
                artifact_path=None if parent == Path(".") else parent.as_posix(),
            )

        # Upload observations as metrics
        metrics_file = run_root / "metrics/observations.csv"
        if metrics_file.is_file():
            batch: list[Metric] = []
            now_ms = int(time.time() * 1000)
            with metrics_file.open(encoding="utf-8", newline="") as stream:
                for row in csv.DictReader(stream):
                    step = int(row.get("processed_tokens", 0))
                    if "objective_loss" in row:
                        batch.append(
                            Metric(
                                "train/objective_loss",
                                float(row["objective_loss"]),
                                now_ms,
                                step,
                            )
                        )
                    if "learning_rate" in row:
                        batch.append(
                            Metric(
                                "train/learning_rate",
                                float(row["learning_rate"]),
                                now_ms,
                                step,
                            )
                        )
                    if "tokens_per_second" in row:
                        batch.append(
                            Metric(
                                "runtime/tokens_per_second",
                                float(row["tokens_per_second"]),
                                now_ms,
                                step,
                            )
                        )
                    if len(batch) >= 750:
                        client.log_batch(run_id, metrics=batch)
                        batch = []
                if batch:
                    client.log_batch(run_id, metrics=batch)

        # Mark as FINISHED only after complete upload
        client.set_terminated(run_id, status="FINISHED")
        return run_id
    except BaseException:
        try:
            client.set_terminated(run_id, status="FAILED")
        except Exception:
            pass
        raise


def _suite(config: dict[str, Any]) -> str:
    tracking = config.get("tracking")
    if isinstance(tracking, dict) and "experiment" in tracking:
        experiment = str(tracking["experiment"])
        suite = experiment.rsplit(".", 1)[-1]
        if suite in {"w2v1", "w2v2"}:
            return suite
    return "w2v2"


def _plan_revision(execution_plan_id: str) -> int:
    match = re.search(r"-r(\d+)$", execution_plan_id)
    if match is None:
        return 1
    return int(match.group(1))


__all__ = [
    "find_matching_finished_run",
    "publish_run_to_mlflow",
]
