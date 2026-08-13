"""Immutable, domain-aware import of DeepScratch legacy MLflow archives."""

from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

from mlflow.entities import Dataset, DatasetInput, InputTag, Metric, Param, ViewType
from mlflow.tracking import MlflowClient

from mlprosection_mlflow.transfer import (
    FORMAT_NAME,
    SUPPORTED_FORMAT_VERSIONS,
    _ordered_runs,
    _path_digest,
    _safe_extract,
)

from ..identity import Variant, Volume
from .namespaces import legacy_namespace


Disposition = Literal[
    "imported",
    "reused-identical",
    "imported-alternate",
    "deferred-running",
    "conflict",
    "rejected-mapping",
]


@dataclass(frozen=True)
class LegacyImportEntry:
    source_run_id: str
    run_key: str | None
    run_type: str | None
    payload_sha256: str
    disposition: Disposition
    target_run_id: str | None = None
    reuse_source_run_id: str | None = None
    reason: str | None = None


@dataclass(frozen=True)
class LegacyImportPlan:
    archive: str
    archive_sha256: str
    source_experiment_name: str
    source_experiment_id: str
    target_experiment_name: str
    entries: tuple[LegacyImportEntry, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            **{key: value for key, value in asdict(self).items() if key != "entries"},
            "entries": [asdict(entry) for entry in self.entries],
        }


_IDENTITY_TAGS = {
    "run.type",
    "run.key",
    "condition.key",
    "condition.group.key",
    "atomic_run.id",
    "condition.id",
    "experiment.id",
    "experiment.ids",
    "execution_group.id",
    "recipe.id",
    "structure.signature",
    "master_seed",
    "seed",
    "protocol.version",
    "implementation.variant",
    "result.schema.name",
    "result.schema.version",
    "mlflow.parentRunId",
    "parent.mlflow_run_id",
}


def inspect_archive(
    archive: str | Path,
    *,
    volume: Volume,
    variant: Variant,
    client: MlflowClient | None = None,
) -> LegacyImportPlan:
    """Preflight every archive run without changing the destination."""
    archive_path = Path(archive).expanduser().resolve()
    archive_sha256 = _path_digest(archive_path)
    target_name = legacy_namespace(volume, variant)
    with tempfile.TemporaryDirectory(prefix="deepscratch-legacy-preflight-") as temp:
        root = Path(temp)
        _safe_extract(archive_path, root)
        manifest = _load_manifest(root)
        source = manifest["experiment"]
        source_name = str(source["name"])
        if source_name != target_name:
            raise ValueError(
                "rejected-mapping: archive source experiment "
                f"{source_name!r} does not match {volume.value}/{variant.value} "
                f"legacy namespace {target_name!r}"
            )
        payloads = {
            str(run["original_run_id"]): _payload_sha256(
                source, run, root / "artifacts" / str(run["original_run_id"])
            )
            for run in manifest["runs"]
        }
        _validate_parent_graph(manifest["runs"])
        entries = _plan_entries(
            client,
            target_name,
            manifest["runs"],
            payloads,
            source,
        )
        return LegacyImportPlan(
            archive=str(archive_path),
            archive_sha256=archive_sha256,
            source_experiment_name=source_name,
            source_experiment_id=str(source["original_experiment_id"]),
            target_experiment_name=target_name,
            entries=tuple(entries),
        )


def import_legacy_archive(
    tracking_uri: str,
    archive: str | Path,
    *,
    volume: Volume,
    variant: Variant,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Append immutable runs to their historical namespace.

    Existing experiments and runs are read-only. Only runs created by this call
    receive transfer metadata.
    """
    client = MlflowClient(tracking_uri=tracking_uri)
    plan = inspect_archive(archive, volume=volume, variant=variant, client=client)
    if dry_run:
        return plan.to_dict()

    archive_path = Path(archive).expanduser().resolve()
    with tempfile.TemporaryDirectory(prefix="deepscratch-legacy-import-") as temp:
        root = Path(temp)
        _safe_extract(archive_path, root)
        manifest = _load_manifest(root)
        source = manifest["experiment"]
        experiment = client.get_experiment_by_name(plan.target_experiment_name)
        if experiment is None:
            experiment_id = client.create_experiment(
                plan.target_experiment_name,
                tags={str(key): str(value) for key, value in source.get("tags", {}).items()},
            )
        else:
            experiment_id = experiment.experiment_id

        planned = {entry.source_run_id: entry for entry in plan.entries}
        run_id_map: dict[str, str] = {}
        results: list[LegacyImportEntry] = []
        for run in _ordered_runs(manifest["runs"]):
            source_run_id = str(run["original_run_id"])
            entry = planned[source_run_id]
            if entry.disposition in {"reused-identical", "deferred-running"}:
                reused_id = entry.target_run_id
                if reused_id is None and entry.reuse_source_run_id is not None:
                    reused_id = run_id_map.get(entry.reuse_source_run_id)
                if reused_id is not None:
                    run_id_map[source_run_id] = reused_id
                results.append(entry)
                continue
            parent_source_id = run.get("tags", {}).get("mlflow.parentRunId")
            if parent_source_id and parent_source_id not in run_id_map:
                results.append(LegacyImportEntry(
                    source_run_id=source_run_id,
                    run_key=entry.run_key,
                    run_type=entry.run_type,
                    payload_sha256=entry.payload_sha256,
                    disposition="conflict",
                    reason=f"parent was not imported or reused: {parent_source_id}",
                ))
                continue
            target_run_id = _create_run(
                client,
                experiment_id,
                run,
                root / "artifacts" / source_run_id,
                source_experiment=source,
                archive_sha256=plan.archive_sha256,
                payload_sha256=entry.payload_sha256,
                disposition=entry.disposition,
                run_id_map=run_id_map,
            )
            run_id_map[source_run_id] = target_run_id
            results.append(LegacyImportEntry(
                **{**asdict(entry), "target_run_id": target_run_id}
            ))

        _verify_new_parent_links(client, results, manifest["runs"], run_id_map)
    return {
        **{key: value for key, value in plan.to_dict().items() if key != "entries"},
        "experiment_id": experiment_id,
        "entries": [asdict(entry) for entry in results],
        "run_id_map": run_id_map,
    }


def _load_manifest(root: Path) -> dict[str, Any]:
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("invalid MLflow transfer archive manifest") from exc
    if (
        manifest.get("format") != FORMAT_NAME
        or manifest.get("version") not in SUPPORTED_FORMAT_VERSIONS
    ):
        raise ValueError("unsupported MLflow transfer archive")
    if not isinstance(manifest.get("runs"), list):
        raise ValueError("archive manifest runs must be a list")
    return manifest


def _payload_sha256(
    source_experiment: dict[str, Any],
    run: dict[str, Any],
    artifact_dir: Path,
) -> str:
    artifacts = []
    if artifact_dir.is_dir():
        for path in sorted(item for item in artifact_dir.rglob("*") if item.is_file()):
            artifacts.append({
                "path": path.relative_to(artifact_dir).as_posix(),
                "sha256": _path_digest(path),
            })
    checkpoint_inventory = run.get("checkpoint_inventory", [])
    for item in checkpoint_inventory:
        if item.get("status") == "present":
            relative = str(item.get("artifact_path", ""))
            match = next((artifact for artifact in artifacts if artifact["path"] == relative), None)
            if match is None or match["sha256"] != item.get("digest"):
                raise ValueError(
                    f"checkpoint digest conflict in archive for "
                    f"{run.get('original_run_id')}/{item.get('role')}"
                )
    tags = {
        str(key): str(value)
        for key, value in run.get("tags", {}).items()
        if key in _IDENTITY_TAGS
    }
    payload = {
        "source_experiment_name": str(source_experiment["name"]),
        "source_experiment_id": str(source_experiment["original_experiment_id"]),
        "original_run_id": str(run["original_run_id"]),
        "params": dict(sorted(run.get("params", {}).items())),
        "metrics": sorted(
            run.get("metrics", []),
            key=lambda item: (
                str(item.get("key")), int(item.get("step", 0)),
                int(item.get("timestamp", 0)), float(item.get("value", 0.0)),
            ),
        ),
        "tags": dict(sorted(tags.items())),
        "artifacts": artifacts,
        "checkpoint_inventory": checkpoint_inventory,
        "dataset_inputs": run.get("dataset_inputs", []),
        "status": run.get("info", {}).get("status"),
        "start_time": run.get("info", {}).get("start_time"),
        "end_time": run.get("info", {}).get("end_time"),
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=True
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _plan_entries(
    client: MlflowClient | None,
    target_name: str,
    runs: Sequence[dict[str, Any]],
    payloads: dict[str, str],
    source_experiment: dict[str, Any],
) -> list[LegacyImportEntry]:
    experiment = None if client is None else client.get_experiment_by_name(target_name)
    entries: list[LegacyImportEntry] = []
    planned_run_keys: dict[str, str] = {}
    planned_parent_groups: dict[str, str] = {}
    for run in _ordered_runs(runs):
        source_run_id = str(run["original_run_id"])
        tags = run.get("tags", {})
        run_key = tags.get("run.key")
        run_type = tags.get("run.type")
        payload = payloads[source_run_id]
        matches = []
        if experiment is not None and client is not None:
            matches = _matching_runs(client, experiment.experiment_id, run)
        identical = next(
            (candidate for candidate in matches if candidate.data.tags.get("transfer.payload.sha256") == payload),
            None,
        )
        if identical is None and client is not None:
            identical = next(
                (
                    candidate
                    for candidate in matches
                    if _existing_payload_sha256(
                        client,
                        candidate,
                        source_experiment,
                        run.get("checkpoint_inventory", []),
                    ) == payload
                ),
                None,
            )
        running = next((candidate for candidate in matches if candidate.info.status == "RUNNING"), None)
        reuse_source_run_id = None
        if run_type == "seed_trial" and running is not None:
            disposition: Disposition = "deferred-running"
            target_run_id = running.info.run_id
            reason = "target run with the same run.key is RUNNING"
        elif identical is not None:
            disposition = "reused-identical"
            target_run_id = identical.info.run_id
            reason = None
        elif run_type == "condition_parent" and matches:
            # Parent identity is the group key. Reuse it without modifying it;
            # child payload identity remains independently auditable.
            disposition = "reused-identical"
            target_run_id = matches[0].info.run_id
            reason = "reused compatible condition parent"
        elif (
            run_type == "condition_parent"
            and tags.get("condition.group.key") in planned_parent_groups
        ):
            disposition = "reused-identical"
            target_run_id = None
            reuse_source_run_id = planned_parent_groups[str(tags["condition.group.key"])]
            reason = "reused compatible condition parent from this archive"
        elif run_type == "seed_trial" and run_key in planned_run_keys:
            disposition = "imported-alternate"
            target_run_id = None
            reason = "same run.key occurs earlier in this archive with a different payload"
        elif matches:
            disposition = "imported-alternate"
            target_run_id = None
            reason = "same logical identity has a different payload"
        else:
            disposition = "imported"
            target_run_id = None
            reason = None
        if disposition in {"imported", "imported-alternate"}:
            if run_type == "seed_trial" and run_key:
                planned_run_keys.setdefault(str(run_key), source_run_id)
            if run_type == "condition_parent" and tags.get("condition.group.key"):
                planned_parent_groups.setdefault(
                    str(tags["condition.group.key"]), source_run_id
                )
        entries.append(LegacyImportEntry(
            source_run_id=source_run_id,
            run_key=None if run_key is None else str(run_key),
            run_type=None if run_type is None else str(run_type),
            payload_sha256=payload,
            disposition=disposition,
            target_run_id=target_run_id,
            reuse_source_run_id=reuse_source_run_id,
            reason=reason,
        ))
    return entries


def _matching_runs(client: MlflowClient, experiment_id: str, run: dict[str, Any]) -> list[Any]:
    tags = run.get("tags", {})
    run_type = tags.get("run.type")
    if run_type == "seed_trial" and tags.get("run.key"):
        expression = (
            "tags.`run.type` = 'seed_trial' AND tags.`run.key` = "
            f"'{_filter_value(str(tags['run.key']))}'"
        )
    elif run_type == "condition_parent" and tags.get("condition.group.key"):
        expression = (
            "tags.`run.type` = 'condition_parent' AND tags.`condition.group.key` = "
            f"'{_filter_value(str(tags['condition.group.key']))}'"
        )
    else:
        expression = (
            "tags.`transfer.source.run_id` = "
            f"'{_filter_value(str(run['original_run_id']))}'"
        )
    return list(client.search_runs(
        [experiment_id],
        filter_string=expression,
        run_view_type=ViewType.ALL,
        order_by=["attributes.start_time ASC"],
        max_results=10_000,
    ))


def _filter_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'")


def _existing_payload_sha256(
    client: MlflowClient,
    candidate: Any,
    source_experiment: dict[str, Any],
    checkpoint_inventory: Sequence[dict[str, Any]],
) -> str:
    """Compute an existing run fingerprint without modifying the run."""
    metrics = [
        {
            "key": metric.key,
            "value": metric.value,
            "timestamp": metric.timestamp,
            "step": metric.step,
            "model_id": metric.model_id,
            "dataset_name": metric.dataset_name,
            "dataset_digest": metric.dataset_digest,
        }
        for key in sorted(candidate.data.metrics)
        for metric in client.get_metric_history(candidate.info.run_id, key)
    ]
    inputs = candidate.inputs
    dataset_inputs = (
        [] if inputs is None
        else [item.to_dictionary() for item in inputs.dataset_inputs]
    )
    record = {
        "original_run_id": candidate.info.run_id,
        "info": {
            "status": candidate.info.status,
            "start_time": candidate.info.start_time,
            "end_time": candidate.info.end_time,
        },
        "params": dict(candidate.data.params),
        "tags": dict(candidate.data.tags),
        "metrics": metrics,
        "dataset_inputs": dataset_inputs,
        "checkpoint_inventory": list(checkpoint_inventory),
    }
    with tempfile.TemporaryDirectory(prefix="deepscratch-existing-payload-") as temp:
        artifact_root = Path(temp)
        try:
            downloaded = Path(
                client.download_artifacts(candidate.info.run_id, "", str(artifact_root))
            )
        except Exception:
            if client.list_artifacts(candidate.info.run_id):
                raise
            downloaded = artifact_root
        return _payload_sha256(source_experiment, record, downloaded)


def _create_run(
    client: MlflowClient,
    experiment_id: str,
    run: dict[str, Any],
    artifact_dir: Path,
    *,
    source_experiment: dict[str, Any],
    archive_sha256: str,
    payload_sha256: str,
    disposition: Disposition,
    run_id_map: dict[str, str],
) -> str:
    source_run_id = str(run["original_run_id"])
    info = run["info"]
    tags = {str(key): str(value) for key, value in run.get("tags", {}).items()}
    for parent_key in ("mlflow.parentRunId", "parent.mlflow_run_id"):
        if tags.get(parent_key) in run_id_map:
            tags[parent_key] = run_id_map[tags[parent_key]]
    instance_key = hashlib.sha256(
        f"{experiment_id}\0{source_run_id}\0{payload_sha256}".encode()
    ).hexdigest()
    tags.update({
        "transfer.source.experiment_name": str(source_experiment["name"]),
        "transfer.source.experiment_id": str(source_experiment["original_experiment_id"]),
        "transfer.source.run_id": source_run_id,
        "transfer.archive.sha256": archive_sha256,
        "transfer.payload.sha256": payload_sha256,
        "transfer.import.instance_key": instance_key,
        "transfer.import.disposition": disposition,
        "transfer.destination.experiment_id": str(experiment_id),
        "transfer.destination.experiment_name": str(source_experiment["name"]),
    })
    created = client.create_run(
        experiment_id,
        start_time=info.get("start_time"),
        run_name=info.get("run_name"),
        tags=tags,
    )
    run_id = created.info.run_id
    params = [Param(str(key), str(value)) for key, value in run.get("params", {}).items()]
    metrics = [Metric(
        str(item["key"]), float(item["value"]), int(item["timestamp"]), int(item["step"]),
        model_id=item.get("model_id"), dataset_name=item.get("dataset_name"),
        dataset_digest=item.get("dataset_digest"),
    ) for item in run.get("metrics", [])]
    for offset in range(0, len(params), 100):
        client.log_batch(run_id, params=params[offset : offset + 100])
    for offset in range(0, len(metrics), 1000):
        client.log_batch(run_id, metrics=metrics[offset : offset + 1000])
    _log_inputs(client, run_id, run.get("dataset_inputs", []))
    _log_and_verify_artifacts(client, run_id, artifact_dir)
    end_time = info.get("end_time")
    status = str(info.get("status", "FINISHED"))
    if end_time is not None:
        client.set_terminated(run_id, status=status, end_time=int(end_time))
    elif status != "RUNNING":
        client.update_run(run_id, status=status)
    if info.get("lifecycle_stage") == "deleted":
        client.delete_run(run_id)
    return run_id


def _log_inputs(client: MlflowClient, run_id: str, inputs: Sequence[dict[str, Any]]) -> None:
    if not inputs:
        return
    client.log_inputs(run_id, datasets=[
        DatasetInput(
            Dataset.from_dictionary(item["dataset"]),
            [InputTag(key, value) for key, value in item.get("tags", {}).items()],
        )
        for item in inputs
    ])


def _log_and_verify_artifacts(client: MlflowClient, run_id: str, root: Path) -> None:
    if not root.is_dir():
        return
    files = sorted(path for path in root.rglob("*") if path.is_file())
    for source in files:
        parent = source.parent.relative_to(root)
        client.log_artifact(
            run_id,
            str(source),
            artifact_path=None if parent == Path(".") else parent.as_posix(),
        )
    with tempfile.TemporaryDirectory(prefix="deepscratch-artifact-verify-") as temp:
        for source in files:
            relative = source.relative_to(root).as_posix()
            target = Path(client.download_artifacts(run_id, relative, temp))
            if _path_digest(source) != _path_digest(target):
                raise ValueError(f"artifact digest conflict after import: {run_id}/{relative}")


def _validate_parent_graph(runs: Sequence[dict[str, Any]]) -> None:
    source_ids = {str(run["original_run_id"]) for run in runs}
    for run in runs:
        tags = run.get("tags", {})
        parents = {
            str(tags[key])
            for key in ("mlflow.parentRunId", "parent.mlflow_run_id")
            if tags.get(key)
        }
        if len(parents) > 1:
            raise ValueError(f"conflict: inconsistent parent tags for {run['original_run_id']}")
        if parents and next(iter(parents)) not in source_ids:
            raise ValueError(f"conflict: archive parent is absent for {run['original_run_id']}")


def _verify_new_parent_links(
    client: MlflowClient,
    results: Sequence[LegacyImportEntry],
    runs: Sequence[dict[str, Any]],
    run_id_map: dict[str, str],
) -> None:
    source = {str(run["original_run_id"]): run for run in runs}
    for result in results:
        if result.disposition not in {"imported", "imported-alternate"} or not result.target_run_id:
            continue
        old_parent = source[result.source_run_id].get("tags", {}).get("mlflow.parentRunId")
        if not old_parent:
            continue
        expected = run_id_map[str(old_parent)]
        actual = client.get_run(result.target_run_id).data.tags.get("mlflow.parentRunId")
        if actual != expected:
            raise ValueError(
                f"parent link verification failed for {result.target_run_id}: {actual} != {expected}"
            )
