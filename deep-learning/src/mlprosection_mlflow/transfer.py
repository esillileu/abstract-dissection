#!/usr/bin/env python3
"""Export and import MLflow experiments and runs as portable ZIP archives."""

from __future__ import annotations

import json
import os
import platform
import shutil
import socket
import subprocess
import tempfile
import zipfile
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from mlflow.entities import Metric, Param, RunTag, ViewType
from mlflow.tracking import MlflowClient

FORMAT_NAME = "mlprosection.mlflow-transfer"
FORMAT_VERSION = 1
PARENT_TAGS = ("mlflow.parentRunId", "parent.mlflow_run_id")


def _chunks(values: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for offset in range(0, len(values), size):
        yield values[offset : offset + size]


def _experiment(client: MlflowClient, reference: str):
    if reference.isdigit():
        try:
            return client.get_experiment(reference)
        except Exception:
            pass
    experiment = client.get_experiment_by_name(reference)
    if experiment is None:
        raise ValueError(f"MLflow experiment not found: {reference}")
    return experiment


def _finished_runs(client: MlflowClient, experiment_id: str) -> list[Any]:
    runs: list[Any] = []
    page_token: str | None = None
    while True:
        page = client.search_runs(
            [experiment_id],
            filter_string="attributes.status = 'FINISHED'",
            run_view_type=ViewType.ALL,
            max_results=1000,
            page_token=page_token,
        )
        runs.extend(page)
        page_token = page.token
        if not page_token:
            return runs


def _metric_dict(metric: Metric) -> dict[str, Any]:
    return {
        "key": metric.key,
        "value": metric.value,
        "timestamp": metric.timestamp,
        "step": metric.step,
        "model_id": metric.model_id,
        "dataset_name": metric.dataset_name,
        "dataset_digest": metric.dataset_digest,
    }


def _run_dict(client: MlflowClient, run: Any) -> dict[str, Any]:
    histories = [
        _metric_dict(metric)
        for key in sorted(run.data.metrics)
        for metric in client.get_metric_history(run.info.run_id, key)
    ]
    inputs = run.inputs
    dataset_inputs = []
    if inputs is not None:
        dataset_inputs = [item.to_dictionary() for item in inputs.dataset_inputs]
    return {
        "original_run_id": run.info.run_id,
        "info": {
            "status": run.info.status,
            "start_time": run.info.start_time,
            "end_time": run.info.end_time,
            "lifecycle_stage": run.info.lifecycle_stage,
            "run_name": run.info.run_name,
            "user_id": run.info.user_id,
            "artifact_uri": run.info.artifact_uri,
        },
        "params": dict(run.data.params),
        "tags": dict(run.data.tags),
        "metrics": histories,
        "dataset_inputs": dataset_inputs,
    }


def _experiment_dict(experiment: Any) -> dict[str, Any]:
    return {
        "original_experiment_id": experiment.experiment_id,
        "name": experiment.name,
        "artifact_location": experiment.artifact_location,
        "lifecycle_stage": experiment.lifecycle_stage,
        "tags": dict(experiment.tags),
        "creation_time": experiment.creation_time,
        "last_update_time": experiment.last_update_time,
    }


def _download_artifacts(client: MlflowClient, run_id: str, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    try:
        downloaded = Path(client.download_artifacts(run_id, "", str(destination)))
    except Exception as exc:
        # Empty artifact directories are valid and some stores reject a root download.
        if client.list_artifacts(run_id):
            raise RuntimeError(f"failed to export artifacts for run {run_id}") from exc
        return
    if downloaded != destination and downloaded.exists():
        for child in downloaded.iterdir():
            target = destination / child.name
            if child.is_dir():
                shutil.copytree(child, target, dirs_exist_ok=True)
            else:
                shutil.copy2(child, target)


def _write_archive(
    client: MlflowClient,
    experiment: Any,
    runs: Sequence[Any],
    kind: str,
    output: str | Path,
) -> Path:
    output_path = Path(output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mlflow-export-") as temp_dir:
        root = Path(temp_dir)
        manifest = {
            "format": FORMAT_NAME,
            "version": FORMAT_VERSION,
            "kind": kind,
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "experiment": _experiment_dict(experiment),
            "runs": [_run_dict(client, run) for run in runs],
        }
        (root / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=True),
            encoding="utf-8",
        )
        for run in runs:
            _download_artifacts(
                client, run.info.run_id, root / "artifacts" / run.info.run_id
            )
        with zipfile.ZipFile(
            output_path, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
        ) as archive:
            for path in sorted(root.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(root))
    return output_path


def export_run(
    tracking_uri: str, run_id: str, output: str | Path
) -> Path:
    client = MlflowClient(tracking_uri=tracking_uri)
    run = client.get_run(run_id)
    experiment = client.get_experiment(run.info.experiment_id)
    return _write_archive(client, experiment, [run], "run", output)


def export_experiment(
    tracking_uri: str, experiment: str, output: str | Path
) -> Path:
    client = MlflowClient(tracking_uri=tracking_uri)
    source_experiment = _experiment(client, experiment)
    return _write_archive(
        client,
        source_experiment,
        _finished_runs(client, source_experiment.experiment_id),
        "experiment",
        output,
    )


def _safe_extract(archive_path: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive_path) as archive:
        destination_root = destination.resolve()
        for member in archive.infolist():
            member_path = (destination / member.filename).resolve()
            if not member_path.is_relative_to(destination_root):
                raise ValueError(f"unsafe archive member: {member.filename}")
        archive.extractall(destination)


def _ordered_runs(runs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Put parents before children while retaining a stable order for other runs."""
    pending = list(runs)
    included = {run["original_run_id"] for run in runs}
    ordered: list[dict[str, Any]] = []
    emitted: set[str] = set()
    while pending:
        ready = []
        for run in pending:
            parent = run["tags"].get("mlflow.parentRunId")
            if parent not in included or parent in emitted:
                ready.append(run)
        if not ready:
            # Malformed/cyclic parent tags should not prevent data recovery.
            ready = [pending[0]]
        for run in ready:
            pending.remove(run)
            ordered.append(run)
            emitted.add(run["original_run_id"])
    return ordered


def _cpu_model() -> str:
    if Path("/proc/cpuinfo").is_file():
        for line in Path("/proc/cpuinfo").read_text(
            encoding="utf-8", errors="replace"
        ).splitlines():
            if line.lower().startswith(("model name", "hardware")):
                return line.partition(":")[2].strip()
    return platform.processor() or platform.machine() or "unknown"


def _tracking_endpoint(tracking_uri: str) -> str:
    parsed = urlsplit(tracking_uri)
    if parsed.scheme in {"http", "https"} and parsed.hostname:
        endpoint = f"{parsed.scheme}://{parsed.hostname}"
        if parsed.port is not None:
            endpoint += f":{parsed.port}"
        return endpoint
    return parsed.scheme or "local"


def _gpu_tags() -> dict[str, str]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return {"transfer.destination.gpu.count": "0"}

    rows = [row.strip() for row in result.stdout.splitlines() if row.strip()]
    tags = {"transfer.destination.gpu.count": str(len(rows))}
    for position, row in enumerate(rows):
        fields = [field.strip() for field in row.split(",", maxsplit=3)]
        if len(fields) != 4:
            continue
        index, name, memory_mb, driver = fields
        prefix = f"transfer.destination.gpu.{position}"
        tags[f"{prefix}.index"] = index
        tags[f"{prefix}.name"] = name
        tags[f"{prefix}.memory_mb"] = memory_mb
        tags[f"{prefix}.driver"] = driver
    return tags


def _environment_tags(tracking_uri: str) -> dict[str, str]:
    try:
        import psutil

        memory_bytes = str(psutil.virtual_memory().total)
    except (ImportError, OSError):
        memory_bytes = "unknown"
    tags = {
        "transfer.destination.hostname": socket.gethostname(),
        "transfer.destination.platform": platform.platform(),
        "transfer.destination.tracking_endpoint": _tracking_endpoint(tracking_uri),
        "transfer.destination.cpu.model": _cpu_model(),
        "transfer.destination.cpu.logical_count": str(os.cpu_count() or "unknown"),
        "transfer.destination.memory.total_bytes": memory_bytes,
        "transfer.imported_at": datetime.now(timezone.utc).isoformat(),
    }
    tags.update(_gpu_tags())
    return tags


def _restore_run(
    client: MlflowClient,
    experiment_id: str,
    run: dict[str, Any],
    run_id_map: dict[str, str],
    artifact_dir: Path,
    destination_tags: dict[str, str],
) -> str:
    info = run["info"]
    created = client.create_run(
        experiment_id,
        start_time=info["start_time"],
        run_name=info["run_name"],
    )
    new_run_id = created.info.run_id
    run_id_map[run["original_run_id"]] = new_run_id

    params = [Param(key, value) for key, value in run["params"].items()]
    tags_dict = {**run["tags"], **destination_tags}
    for parent_tag in PARENT_TAGS:
        old_parent = tags_dict.get(parent_tag)
        if old_parent in run_id_map:
            tags_dict[parent_tag] = run_id_map[old_parent]
    tags = [RunTag(key, value) for key, value in tags_dict.items()]
    metrics = [
        Metric(
            item["key"],
            item["value"],
            item["timestamp"],
            item["step"],
            model_id=item.get("model_id"),
            dataset_name=item.get("dataset_name"),
            dataset_digest=item.get("dataset_digest"),
        )
        for item in run["metrics"]
    ]

    for batch in _chunks(params, 100):
        client.log_batch(new_run_id, params=batch)
    for batch in _chunks(tags, 100):
        client.log_batch(new_run_id, tags=batch)
    for batch in _chunks(metrics, 1000):
        client.log_batch(new_run_id, metrics=batch)

    dataset_inputs = run.get("dataset_inputs", [])
    if dataset_inputs:
        from mlflow.entities import Dataset, DatasetInput, InputTag

        client.log_inputs(
            new_run_id,
            datasets=[
                DatasetInput(
                    Dataset.from_dictionary(item["dataset"]),
                    [
                        InputTag(key, value)
                        for key, value in item.get("tags", {}).items()
                    ],
                )
                for item in dataset_inputs
            ],
        )
    if artifact_dir.is_dir() and any(artifact_dir.iterdir()):
        client.log_artifacts(new_run_id, str(artifact_dir))

    if info["end_time"] is not None:
        client.set_terminated(
            new_run_id, status=info["status"], end_time=info["end_time"]
        )
    elif info["status"] != created.info.status:
        client.update_run(new_run_id, status=info["status"])
    if info["lifecycle_stage"] == "deleted":
        client.delete_run(new_run_id)
    return new_run_id


def import_archive(
    tracking_uri: str,
    archive: str | Path,
    *,
    experiment_name: str | None = None,
    artifact_location: str | None = None,
    capture_environment: bool = True,
    destination_tags: dict[str, str] | None = None,
    reuse_experiment: bool = False,
) -> dict[str, Any]:
    archive_path = Path(archive).expanduser().resolve()
    client = MlflowClient(tracking_uri=tracking_uri)
    with tempfile.TemporaryDirectory(prefix="mlflow-import-") as temp_dir:
        root = Path(temp_dir)
        _safe_extract(archive_path, root)
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if (
            manifest.get("format") != FORMAT_NAME
            or manifest.get("version") != FORMAT_VERSION
        ):
            raise ValueError("unsupported MLflow transfer archive")

        source_experiment = manifest["experiment"]
        target_name = experiment_name or source_experiment["name"]
        applied_destination_tags = (
            _environment_tags(tracking_uri) if capture_environment else {}
        )
        if destination_tags:
            applied_destination_tags.update(
                {str(key): str(value) for key, value in destination_tags.items()}
            )
        existing_experiment = client.get_experiment_by_name(target_name)
        if existing_experiment is not None and not reuse_experiment:
            raise ValueError(
                f"target experiment already exists: {target_name}; "
                "use --reuse-experiment to append runs or "
                "--experiment-name to choose a new name"
            )
        reused_experiment = existing_experiment is not None
        if existing_experiment is not None:
            experiment_id = existing_experiment.experiment_id
            merged_experiment_tags = {
                **source_experiment["tags"],
                **existing_experiment.tags,
                **applied_destination_tags,
            }
            for key, value in merged_experiment_tags.items():
                client.set_experiment_tag(experiment_id, key, value)
        else:
            experiment_id = client.create_experiment(
                target_name,
                artifact_location=artifact_location,
                tags={**source_experiment["tags"], **applied_destination_tags},
            )
        run_id_map: dict[str, str] = {}
        try:
            for run in _ordered_runs(manifest["runs"]):
                _restore_run(
                    client,
                    experiment_id,
                    run,
                    run_id_map,
                    root / "artifacts" / run["original_run_id"],
                    applied_destination_tags,
                )
            if (
                source_experiment["lifecycle_stage"] == "deleted"
                and not reused_experiment
            ):
                client.delete_experiment(experiment_id)
        except Exception:
            # Keep the partially imported experiment for diagnosis and recovery.
            raise
    return {
        "experiment_id": experiment_id,
        "experiment_name": target_name,
        "reused_experiment": reused_experiment,
        "run_id_map": run_id_map,
    }
