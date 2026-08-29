"""Resolve checkpoint dependencies declared by observation-run configs."""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any


class CheckpointSourceRunNotFound(ValueError):
    """The declared experiment has no matching checkpoint-producing run."""


def resolve_checkpoint_source(
    config: dict[str, object],
    *,
    client: Any | None = None,
) -> Path | None:
    """Resolve a matching-seed source checkpoint and update ``source_path``.

    A concrete source path always wins. Otherwise the source run is selected
    from MLflow by group, atomic run ID, and master seed. Local checkpoint
    generations are preferred; transferred runs fall back to a downloaded
    MLflow checkpoint artifact.
    """
    checkpoint = config.get("checkpoint")
    if not isinstance(checkpoint, dict):
        return None
    explicit = checkpoint.get("source_path") or checkpoint.get("source_checkpoint_path")
    if explicit:
        return Path(str(explicit))

    source_group = checkpoint.get("source_group_id")
    source_atomic = checkpoint.get("source_atomic_run_id")
    if not source_group or not source_atomic:
        return None

    tracking = config.get("tracking", {})
    if not isinstance(tracking, dict) or not bool(tracking.get("enabled", True)):
        raise ValueError("checkpoint source resolution requires MLflow tracking")
    experiment_name = str(tracking.get("experiment", "mlprosection"))
    return resolve_checkpoint_source_in_experiment(
        config,
        experiment_name=experiment_name,
        client=client,
    )


def resolve_checkpoint_source_in_experiment(
    config: dict[str, object],
    *,
    experiment_name: str,
    client: Any | None = None,
) -> Path:
    """Resolve a declared source from one explicit MLflow experiment.

    This lower-level entry point lets callers select one explicit experiment
    without coupling the resolver to a particular namespace.
    """
    checkpoint = config.get("checkpoint")
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint source resolution requires checkpoint config")
    source_group = checkpoint.get("source_group_id")
    source_atomic = checkpoint.get("source_atomic_run_id")
    if not source_group or not source_atomic:
        raise ValueError("checkpoint source resolution requires source coordinates")

    tracking = config.get("tracking", {})
    tracking_uri = (
        os.getenv("REPRO_TRACKING_URI")
        or os.getenv("MLFLOW_TRACKING_URI")
        or os.getenv("MLFLOW_F2_URL")
        or os.getenv("MLFLOW_DLFS_URL")
        or os.getenv("MLFLOW_F1_URL")
        or str(tracking.get("uri", "http://127.0.0.1:5000"))
    )
    if client is None:
        from mlflow.tracking import MlflowClient

        client = MlflowClient(tracking_uri=tracking_uri)
    from .artifact_cache import MlflowArtifactCache

    artifact_cache = MlflowArtifactCache(client, tracking_uri)

    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise CheckpointSourceRunNotFound(
            f"checkpoint source experiment does not exist: {experiment_name}"
        )
    candidates = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=(
            "attributes.status = 'FINISHED' and "
            "tags.`run.type` = 'seed_trial' and "
            f"tags.`execution_group.id` = '{source_group}' and "
            f"tags.`atomic_run.id` = '{source_atomic}'"
        ),
        order_by=["attributes.start_time DESC"],
        max_results=5_000,
    )
    source_protocol = checkpoint.get("source_protocol_version")
    seed = str(config.get("seed", ""))
    source_run = next(
        (
            run
            for run in candidates
            if source_protocol is None
            or run.data.tags.get("protocol.version", "legacy") == str(source_protocol)
            if str(run.data.params.get("seed/master", run.data.params.get("seed", "")))
            == seed
        ),
        None,
    )
    if source_run is None:
        raise CheckpointSourceRunNotFound(
            "matching checkpoint source run is missing: "
            f"{source_group}/{source_atomic} seed={seed}"
        )

    arbitrary_artifact = checkpoint.get("source_artifact_path")
    if arbitrary_artifact:
        artifact_path = str(arbitrary_artifact).strip("/")
        if not artifact_path or ".." in Path(artifact_path).parts:
            raise ValueError(
                f"invalid checkpoint source artifact path: {arbitrary_artifact}"
            )
        try:
            resolved = artifact_cache.get(source_run.info.run_id, artifact_path)
        except Exception as exc:
            raise ValueError(
                "checkpoint source artifact is missing for "
                f"{source_group}/{source_atomic} seed={seed}: {artifact_path}"
            ) from exc
        if not resolved.exists():
            raise ValueError(f"downloaded checkpoint artifact is missing: {resolved}")
        checkpoint["source_path"] = str(resolved)
        return resolved

    source_kind = str(checkpoint.get("source_kind", "latest"))
    role = {"selected": "best", "final": "latest", "latest": "latest"}.get(source_kind)
    if role is None:
        raise ValueError(f"unsupported checkpoint source kind: {source_kind}")

    run_key = source_run.data.tags.get("run.key")
    if run_key:
        local_root = _local_checkpoint_root(source_run, str(run_key))
        pointer = None if local_root is None else local_root / f"{role}.json"
        if pointer is not None and pointer.is_file():
            from repro_core.context.checkpoint import resolve_checkpoint_path

            resolved = resolve_checkpoint_path(pointer)
            if resolved.is_dir():
                checkpoint["source_path"] = str(resolved)
                return resolved

    checkpoint_row = _checkpoint_row(
        client,
        source_run.info.run_id,
        source_kind=source_kind,
        artifact_cache=artifact_cache,
    )
    checkpoint_name = Path(checkpoint_row["path"]).name
    artifact_paths = (
        f"checkpoints/generations/{checkpoint_name}",
        f"checkpoints/{checkpoint_name}",
    )
    resolved = None
    last_error: Exception | None = None
    for artifact_path in artifact_paths:
        try:
            resolved = artifact_cache.get(source_run.info.run_id, artifact_path)
            break
        except Exception as exc:
            last_error = exc
    if resolved is None:
        raise ValueError(
            f"{source_kind} checkpoint payload is missing for "
            f"{source_group}/{source_atomic} seed={seed}; rerun the source "
            "with tracking.upload_eval_checkpoints=true"
        ) from last_error
    if not resolved.is_dir():
        raise ValueError(f"downloaded checkpoint is not a directory: {resolved}")
    checkpoint["source_path"] = str(resolved)
    return resolved


def _checkpoint_row(
    client: Any,
    run_id: str,
    *,
    source_kind: str,
    artifact_cache,
) -> dict[str, str]:
    try:
        path = artifact_cache.get(run_id, "checkpoints.csv")
    except Exception as exc:
        raise ValueError(
            f"checkpoint index is missing for source run {run_id}"
        ) from exc
    with path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    wanted = "selected" if source_kind == "selected" else "latest"
    selected = [row for row in rows if row.get("kind") == wanted and row.get("path")]
    if not selected:
        raise ValueError(
            f"{source_kind} checkpoint is not indexed for source run {run_id}"
        )
    return selected[-1]


def _local_checkpoint_root(source_run: Any, run_key: str) -> Path | None:
    """Resolve canonical staging only when the run advertises its coordinate."""
    from repro_core.context.paths import WorkspacePaths

    tags = source_run.data.tags
    coordinate = (
        tags.get("domain.name"),
        tags.get("suite.name"),
        tags.get("experiment.id"),
        tags.get("implementation.variant"),
    )
    if not all(coordinate):
        return None
    staging = WorkspacePaths.from_environment(Path.cwd()).run_staging(
        domain=str(coordinate[0]),
        suite=str(coordinate[1]),
        study=str(coordinate[2]),
        variant=str(coordinate[3]),
        run_key=run_key,
    )
    return staging / "checkpoints"
