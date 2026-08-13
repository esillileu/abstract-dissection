"""Read retired namespaces and project their coordinates at the boundary."""

from __future__ import annotations

from typing import Any

from mlflow.entities import ViewType

from ..identity import Variant, Volume
from .namespaces import legacy_namespace


def load_legacy_attempts(client, volume: Volume, variant: Variant) -> list[dict[str, object]]:
    namespace = legacy_namespace(volume, variant)
    experiment = client.get_experiment_by_name(namespace)
    if experiment is None:
        return []
    runs = client.search_runs(
        [experiment.experiment_id],
        run_view_type=ViewType.ACTIVE_ONLY,
        order_by=["attributes.start_time DESC"],
        max_results=10_000,
    )
    output = []
    for run in runs:
        tags = run.data.tags
        if tags.get("run.type") != "seed_trial":
            continue
        study = _first(
            tags.get("experiment.id"), tags.get("experiment.ids"),
            run.data.params.get("experiment_id"),
            run.data.params.get("run/experiment_id"),
        )
        condition = _first(
            tags.get("condition.id"), tags.get("atomic_run.id"),
            run.data.params.get("atomic_run_id"),
        )
        if not study or not condition:
            continue
        output.append({
            "run_id": run.info.run_id,
            "namespace": namespace,
            "study_id": study,
            "condition_id": condition,
            "seed": _first(
                tags.get("seed"), tags.get("master_seed"),
                run.data.params.get("seed/master"), run.data.params.get("seed"),
            ) or "single",
            "protocol_version": tags.get("protocol.version", "legacy"),
            "status": str(run.info.status).upper(),
            "start_time": int(run.info.start_time or 0),
            "disposition": tags.get("transfer.import.disposition"),
            "durable_complete": None,
        })
    return output


def _first(*values: Any) -> str:
    for value in values:
        if value is not None and (text := str(value).split(",")[0]):
            return text
    return ""
