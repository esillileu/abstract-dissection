"""Durable MLflow recording for raw DeepScratch profile measurements."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path


def record_profile_result(
    tracking_uri: str,
    *,
    volume: str,
    experiment_id: str,
    variant: str,
    output: Path,
) -> str:
    """Upload a completed profile payload and only then mark it durable."""
    from mlflow import MlflowClient

    if not output.exists():
        raise ValueError(f"profile output does not exist: {output}")
    client = MlflowClient(tracking_uri=tracking_uri)
    namespace = f"deepscratch.{volume}"
    experiment = client.get_experiment_by_name(namespace)
    experiment_id_mlflow = (
        client.create_experiment(
            namespace,
            artifact_location=_local_artifact_location(tracking_uri),
        )
        if experiment is None
        else experiment.experiment_id
    )
    condition_id = f"profile:{experiment_id}:{variant}"
    tags = {
        "domain.name": "deepscratch",
        "deepscratch.volume": volume,
        "implementation.variant": variant,
        "experiment.id": experiment_id,
        "experiment.ids": experiment_id,
        "condition.id": condition_id,
        "atomic_run.id": condition_id,
        "run.type": "profile",
        "result.schema.name": f"{volume}-profile",
        "result.schema.version": "1",
        "protocol.version": "profile-v1",
        "result.durable_complete": "false",
    }
    run = client.create_run(
        str(experiment_id_mlflow),
        tags={**tags, "mlflow.runName": condition_id},
    )
    try:
        inventory = _inventory(output)
        with tempfile.TemporaryDirectory(prefix="deepscratch-profile-") as temporary:
            manifest = Path(temporary) / "profile_manifest.json"
            manifest.write_text(
                json.dumps(
                    {"schema_version": 1, "files": inventory},
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            if output.is_dir():
                client.log_artifacts(run.info.run_id, str(output), "profile/raw")
            else:
                client.log_artifact(run.info.run_id, str(output), "profile/raw")
            client.log_artifact(run.info.run_id, str(manifest), "profile")
        client.set_tag(run.info.run_id, "profile.file_count", str(len(inventory)))
        client.set_tag(run.info.run_id, "result.durable_complete", "true")
        client.set_terminated(run.info.run_id, status="FINISHED")
    except Exception:
        client.set_tag(run.info.run_id, "result.durable_complete", "false")
        client.set_terminated(run.info.run_id, status="FAILED")
        raise
    return run.info.run_id


def _local_artifact_location(tracking_uri: str) -> str | None:
    """Keep SQLite-backed test/dev artifacts beside the database, never in cwd."""
    prefix = "sqlite:///"
    if not tracking_uri.startswith(prefix):
        return None
    database = Path(tracking_uri.removeprefix(prefix)).resolve()
    return (database.parent / "artifacts").as_uri()


def _inventory(output: Path) -> list[dict[str, object]]:
    files = [output] if output.is_file() else sorted(
        path for path in output.rglob("*") if path.is_file()
    )
    root = output.parent if output.is_file() else output
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "size": path.stat().st_size,
            "sha256": _digest(path),
        }
        for path in files
    ]


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
