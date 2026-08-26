#!/usr/bin/env python3
"""Export and import MLflow experiments and runs as portable ZIP archives."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import tempfile
import zipfile
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from mlflow.entities import Metric, Param, RunTag, ViewType
from mlflow.tracking import MlflowClient
from tqdm.auto import tqdm

from .run_relationships import PARENT_TAGS, reconcile_parent_links

FORMAT_NAME = "mlprosection.mlflow-transfer"
FORMAT_VERSION = 2
SUPPORTED_FORMAT_VERSIONS = {1, 2}


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
    with tqdm(
        desc="Finding finished runs",
        unit="run",
        disable=None,
    ) as progress:
        while True:
            page = client.search_runs(
                [experiment_id],
                filter_string="attributes.status = 'FINISHED'",
                run_view_type=ViewType.ALL,
                max_results=1000,
                page_token=page_token,
            )
            runs.extend(page)
            progress.update(len(page))
            page_token = page.token
            if not page_token:
                return runs


def _with_parent_dependencies(
    client: MlflowClient,
    runs: Sequence[Any],
) -> list[Any]:
    """Include every available parent referenced by the selected runs."""
    selected = {run.info.run_id: run for run in runs}
    pending = list(runs)
    while pending:
        run = pending.pop()
        parent_id = run.data.tags.get("mlflow.parentRunId")
        if not parent_id or parent_id in selected:
            continue
        try:
            parent = client.get_run(parent_id)
        except Exception:
            # Preserve exportability of legacy runs with already-dangling tags.
            continue
        if parent.info.experiment_id != run.info.experiment_id:
            continue
        selected[parent.info.run_id] = parent
        pending.append(parent)
    return list(selected.values())


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
        "checkpoint_inventory": [],
    }


def _path_digest(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_file():
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    for item in sorted(value for value in path.rglob("*") if value.is_file()):
        digest.update(item.relative_to(path).as_posix().encode())
        with item.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _checkpoint_inventory(
    run: dict[str, Any], artifact_dir: Path
) -> list[dict[str, Any]]:
    """Materialize remote or local-only checkpoint roles into an archive run."""
    checkpoint_dir = artifact_dir / "checkpoints"
    manifest = _read_json(checkpoint_dir / "checkpoint_manifest.json") or {}
    local_root_value = manifest.get("local_root")
    local_root = Path(str(local_root_value)) if local_root_value else None
    not_applicable = run["tags"].get("run.type") != "seed_trial" or str(
        run["tags"].get("execution_group.id", "")
    ).upper().startswith("GO")
    inventory: list[dict[str, Any]] = []
    for role in ("latest", "best"):
        if not_applicable:
            inventory.append({"role": role, "status": "not_applicable"})
            continue
        pointer = _read_json(checkpoint_dir / f"{role}.json")
        candidate: Path | None = None
        if pointer and pointer.get("path"):
            possible = checkpoint_dir / str(pointer["path"])
            if possible.exists():
                candidate = possible
        item = manifest.get(role)
        if role == "latest" and not item:
            item = manifest.get("final")
        if candidate is None and isinstance(item, dict) and item.get("path"):
            possible = Path(str(item["path"]))
            if not possible.is_absolute() and local_root is not None:
                possible = local_root / possible
            if possible.exists():
                candidate = possible
        if candidate is None and role == "latest" and local_root is not None:
            legacy = local_root / "final.npz"
            if legacy.exists():
                candidate = legacy
        if candidate is None:
            inventory.append(
                {
                    "role": role,
                    "status": "missing",
                    "reason": "checkpoint absent from MLflow and export machine",
                }
            )
            continue
        digest = _path_digest(candidate)
        destination = checkpoint_dir / "generations" / candidate.name
        if candidate.is_dir():
            if candidate.resolve() != destination.resolve():
                shutil.copytree(candidate, destination, dirs_exist_ok=True)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            if candidate.resolve() != destination.resolve():
                shutil.copy2(candidate, destination)
        relative = destination.relative_to(artifact_dir).as_posix()
        pointer_path = destination.relative_to(checkpoint_dir).as_posix()
        (checkpoint_dir / f"{role}.json").write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "role": role,
                    "path": pointer_path,
                    "sha256": digest,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        inventory.append(
            {
                "role": role,
                "status": "present",
                "artifact_path": relative,
                "digest": digest,
            }
        )
    return inventory


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


@contextmanager
def _mlflow_artifact_progress_disabled() -> Iterator[None]:
    variable = "MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR"
    previous = os.environ.get(variable)
    os.environ[variable] = "false"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(variable, None)
        else:
            os.environ[variable] = previous


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
        run_records = [
            _run_dict(client, run)
            for run in tqdm(
                runs,
                desc="Collecting run data",
                unit="run",
                disable=None,
            )
        ]
        with _mlflow_artifact_progress_disabled():
            for run in tqdm(
                runs,
                desc="Downloading artifacts",
                unit="run",
                disable=None,
            ):
                _download_artifacts(
                    client, run.info.run_id, root / "artifacts" / run.info.run_id
                )
        for run_record in run_records:
            run_record["checkpoint_inventory"] = _checkpoint_inventory(
                run_record,
                root / "artifacts" / run_record["original_run_id"],
            )
        manifest = {
            "format": FORMAT_NAME,
            "version": FORMAT_VERSION,
            "kind": kind,
            "exported_at": datetime.now(UTC).isoformat(),
            "experiment": _experiment_dict(experiment),
            "runs": run_records,
        }
        (root / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=True),
            encoding="utf-8",
        )
        archive_files = sorted(path for path in root.rglob("*") if path.is_file())
        with zipfile.ZipFile(
            output_path, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
        ) as archive:
            for path in tqdm(
                archive_files,
                desc="Creating archive",
                unit="file",
                disable=None,
            ):
                archive.write(path, path.relative_to(root))
    return output_path


def export_run(tracking_uri: str, run_id: str, output: str | Path) -> Path:
    client = MlflowClient(tracking_uri=tracking_uri)
    run = client.get_run(run_id)
    experiment = client.get_experiment(run.info.experiment_id)
    return _write_archive(
        client,
        experiment,
        _with_parent_dependencies(client, [run]),
        "run",
        output,
    )


def export_experiment(tracking_uri: str, experiment: str, output: str | Path) -> Path:
    client = MlflowClient(tracking_uri=tracking_uri)
    source_experiment = _experiment(client, experiment)
    return _write_archive(
        client,
        source_experiment,
        _with_parent_dependencies(
            client,
            _finished_runs(client, source_experiment.experiment_id),
        ),
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
        for line in (
            Path("/proc/cpuinfo")
            .read_text(encoding="utf-8", errors="replace")
            .splitlines()
        ):
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
        "transfer.imported_at": datetime.now(UTC).isoformat(),
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
    existing = _find_existing_run(client, experiment_id, run)
    if existing is None:
        created = client.create_run(
            experiment_id,
            start_time=info["start_time"],
            run_name=info["run_name"],
            tags=_identity_tags(run),
        )
        new_run_id = created.info.run_id
    else:
        created = existing
        new_run_id = existing.info.run_id
        if existing.info.lifecycle_stage == "deleted":
            # MLflow rejects metadata and artifact writes to deleted runs.
            # Restore temporarily; the source lifecycle is reapplied below.
            client.restore_run(new_run_id)
            created = client.get_run(new_run_id)
    run_id_map[run["original_run_id"]] = new_run_id

    current = client.get_run(new_run_id)
    for key, value in run["params"].items():
        previous = current.data.params.get(key)
        if previous is not None and previous != value:
            raise ValueError(
                f"param conflict for {new_run_id}/{key}: {previous!r} != {value!r}"
            )
    params = [
        Param(key, value)
        for key, value in run["params"].items()
        if key not in current.data.params
    ]
    tags_dict = {**run["tags"], **destination_tags}
    tags_dict.update(_identity_tags(run))
    for parent_tag in PARENT_TAGS:
        old_parent = tags_dict.get(parent_tag)
        if old_parent in run_id_map:
            tags_dict[parent_tag] = run_id_map[old_parent]
    tags = [RunTag(key, value) for key, value in tags_dict.items()]
    source_metrics = [
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
    existing_metric_tuples: dict[tuple[str, int, int], float] = {}
    for key in current.data.metrics:
        for metric in client.get_metric_history(new_run_id, key):
            existing_metric_tuples[(key, metric.step, metric.timestamp)] = metric.value
    metrics = []
    for metric in source_metrics:
        identity = (metric.key, metric.step, metric.timestamp)
        if identity in existing_metric_tuples:
            if existing_metric_tuples[identity] != metric.value:
                raise ValueError(
                    f"metric conflict for {new_run_id}/{identity}: "
                    f"{existing_metric_tuples[identity]} != {metric.value}"
                )
            continue
        metrics.append(metric)

    for batch in _chunks(params, 100):
        client.log_batch(new_run_id, params=batch)
    for batch in _chunks(tags, 100):
        client.log_batch(new_run_id, tags=batch)
    for batch in _chunks(metrics, 1000):
        client.log_batch(new_run_id, metrics=batch)

    dataset_inputs = run.get("dataset_inputs", [])
    current_inputs = client.get_run(new_run_id).inputs
    existing_inputs = (
        []
        if current_inputs is None
        else [item.to_dictionary() for item in current_inputs.dataset_inputs]
    )
    if existing_inputs and existing_inputs != dataset_inputs:
        raise ValueError(f"dataset input conflict for {new_run_id}")
    if dataset_inputs and not existing_inputs:
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
        _supplement_artifacts(client, new_run_id, artifact_dir)

    if info["end_time"] is not None:
        client.set_terminated(
            new_run_id, status=info["status"], end_time=info["end_time"]
        )
    elif info["status"] != created.info.status:
        client.update_run(new_run_id, status=info["status"])
    return new_run_id


def _identity_tags(run: dict[str, Any]) -> dict[str, str]:
    tags = run["tags"]
    run_type = tags.get("run.type")
    if run_type == "seed_trial" and tags.get("run.key"):
        return {"run.key": str(tags["run.key"])}
    if run_type == "condition_parent" and tags.get("condition.group.key"):
        return {"condition.group.key": str(tags["condition.group.key"])}
    return {
        "transfer.source.run_id": str(run["original_run_id"]),
        "transfer.source.experiment_id": str(run.get("source_experiment_id", "")),
        "transfer.source.experiment_name": str(run.get("source_experiment_name", "")),
    }


def _find_existing_run(
    client: MlflowClient,
    experiment_id: str,
    run: dict[str, Any],
):
    identity = _identity_tags(run)
    filters = []
    if "run.key" in identity:
        filters = [
            "tags.`run.type` = 'seed_trial'",
            f"tags.`run.key` = '{identity['run.key']}'",
        ]
    elif "condition.group.key" in identity:
        filters = [
            "tags.`run.type` = 'condition_parent'",
            f"tags.`condition.group.key` = '{identity['condition.group.key']}'",
        ]
    else:
        filters = [
            f"tags.`transfer.source.run_id` = '{identity['transfer.source.run_id']}'",
            "tags.`transfer.source.experiment_id` = "
            f"'{identity['transfer.source.experiment_id']}'",
            "tags.`transfer.source.experiment_name` = "
            f"'{identity['transfer.source.experiment_name']}'",
        ]
    matches = client.search_runs(
        [experiment_id],
        filter_string=" AND ".join(filters),
        run_view_type=ViewType.ALL,
        order_by=["attributes.start_time DESC"],
        max_results=2,
    )
    if len(matches) > 1:
        raise ValueError(
            f"multiple target runs already match import identity: {identity}"
        )
    return matches[0] if matches else None


def _supplement_artifacts(
    client: MlflowClient,
    run_id: str,
    artifact_dir: Path,
) -> None:
    existing = {
        item.path for item in _walk_artifacts(client, run_id) if not item.is_dir
    }
    with tempfile.TemporaryDirectory(prefix="mlflow-artifact-verify-") as temp:
        for source in sorted(
            path for path in artifact_dir.rglob("*") if path.is_file()
        ):
            relative = source.relative_to(artifact_dir).as_posix()
            if relative in existing:
                target = Path(client.download_artifacts(run_id, relative, temp))
                if not _matching_artifact(target, source, relative):
                    raise ValueError(
                        f"artifact digest conflict for {run_id}/{relative}"
                    )
                continue
            parent = source.parent.relative_to(artifact_dir)
            client.log_artifact(
                run_id,
                str(source),
                artifact_path=None if parent == Path(".") else parent.as_posix(),
            )


def _matching_artifact(target: Path, source: Path, relative: str) -> bool:
    if _path_digest(target) == _path_digest(source):
        return True
    if Path(relative).name != "checkpoint_manifest.json":
        return False
    target_manifest = _read_json(target)
    source_manifest = _read_json(source)
    return (
        target_manifest is not None
        and source_manifest is not None
        and target_manifest == source_manifest
    )


def _walk_artifacts(client: MlflowClient, run_id: str):
    pending = [""]
    while pending:
        parent = pending.pop()
        for item in client.list_artifacts(run_id, parent):
            yield item
            if item.is_dir:
                pending.append(item.path)


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
            or manifest.get("version") not in SUPPORTED_FORMAT_VERSIONS
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
        lifecycle_stages_by_run: dict[str, set[str]] = {}
        relationship_entries: list[dict[str, Any]] = []
        source_identity = str(source_experiment["original_experiment_id"])
        for run in manifest["runs"]:
            run["source_experiment_id"] = source_identity
            run["source_experiment_name"] = str(source_experiment["name"])
        lifecycle_stages_by_identity: dict[frozenset[tuple[str, str]], set[str]] = {}
        for run in manifest["runs"]:
            identity = frozenset(_identity_tags(run).items())
            lifecycle_stages_by_identity.setdefault(identity, set()).add(
                str(run["info"]["lifecycle_stage"])
            )
        try:
            for run in _ordered_runs(manifest["runs"]):
                identity = frozenset(_identity_tags(run).items())
                try:
                    target_run_id = _restore_run(
                        client,
                        experiment_id,
                        run,
                        run_id_map,
                        root / "artifacts" / run["original_run_id"],
                        applied_destination_tags,
                    )
                finally:
                    mapped_run_id = run_id_map.get(run["original_run_id"])
                    if mapped_run_id is not None:
                        lifecycle_stages_by_run.setdefault(mapped_run_id, set()).update(
                            lifecycle_stages_by_identity[identity]
                        )
                _verify_checkpoint_inventory(
                    client,
                    target_run_id,
                    run.get("checkpoint_inventory", []),
                )
        finally:
            for target_run_id, lifecycle_stages in lifecycle_stages_by_run.items():
                # Multiple source runs can intentionally collapse onto one logical
                # target identity. Keep that target active when any source is active;
                # otherwise apply the shared deleted lifecycle only after all writes
                # and checkpoint verification have completed.
                if lifecycle_stages == {"deleted"}:
                    client.delete_run(target_run_id)
        touched_group_keys = {
            str(run["tags"]["condition.group.key"])
            for run in manifest["runs"]
            if run["tags"].get("condition.group.key")
        }
        relationship_entries = reconcile_parent_links(
            client,
            experiment_id,
            group_keys=touched_group_keys,
            apply=True,
        )
        if source_experiment["lifecycle_stage"] == "deleted" and not reused_experiment:
            client.delete_experiment(experiment_id)
    return {
        "experiment_id": experiment_id,
        "experiment_name": target_name,
        "reused_experiment": reused_experiment,
        "run_id_map": run_id_map,
        "relationship_repairs": relationship_entries,
    }


def _verify_checkpoint_inventory(
    client: MlflowClient,
    run_id: str,
    inventory: Sequence[dict[str, Any]],
) -> None:
    for item in inventory:
        role = str(item.get("role"))
        status = item.get("status")
        if status == "missing":
            client.set_tag(run_id, f"checkpoint.{role}.status", "missing")
            client.set_tag(
                run_id,
                f"checkpoint.{role}.missing_reason",
                str(item.get("reason", "missing in source archive")),
            )
            continue
        if status == "not_applicable":
            client.set_tag(run_id, f"checkpoint.{role}.status", "not_applicable")
            continue
        if status != "present":
            raise ValueError(f"invalid checkpoint inventory status: {status!r}")
        artifact_path = str(item["artifact_path"])
        with tempfile.TemporaryDirectory(prefix="mlflow-checkpoint-verify-") as temp:
            downloaded = Path(client.download_artifacts(run_id, artifact_path, temp))
            digest = _path_digest(downloaded)
        if digest != item["digest"]:
            raise ValueError(
                f"checkpoint digest mismatch after import for "
                f"{run_id}/{role}: {digest} != {item['digest']}"
            )
        client.set_tag(run_id, f"checkpoint.{role}.status", "present")
        client.set_tag(run_id, f"checkpoint.{role}.sha256", str(item["digest"]))
