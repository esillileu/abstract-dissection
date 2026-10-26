"""MLflow publication and discovery contracts for F2 Word2Vec runs."""

from __future__ import annotations

import csv
import re
import time
from pathlib import Path
from typing import Any

from mlflow import MlflowClient
from mlflow.entities import Metric, Run

from f2.run_identity import RunIdentity
from repro_mlflow.schema_v1 import write_result_manifest


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
        # Write local result manifest
        write_result_manifest(
            run_root,
            exclude_dirs=("preparation",)
            if include_checkpoints
            else ("checkpoints", "preparation"),
        )

        # Upload artifacts
        for path in sorted(item for item in run_root.rglob("*") if item.is_file()):
            rel = path.relative_to(run_root)
            if "preparation" in rel.parts:
                continue
            if "checkpoints" in rel.parts and any(
                part.startswith(".") or part.startswith("epoch-") for part in rel.parts
            ):
                continue
            if not include_checkpoints and "checkpoints" in rel.parts:
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
