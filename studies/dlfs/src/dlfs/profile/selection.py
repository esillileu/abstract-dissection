"""Canonical MLflow selection for durable performance-profile attempts."""

from __future__ import annotations

from mlflow.entities import ViewType


def latest_profile_runs(
    client,
    *,
    experiment_name: str,
    study_id: str,
    device: str | None = None,
    timing_source: str | None = None,
    schema_name: str | None = None,
    protocol_version: str | None = None,
):
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return []
    candidates = client.search_runs(
        [experiment.experiment_id],
        filter_string="tags.`run.type` = 'profile'",
        run_view_type=ViewType.ACTIVE_ONLY,
        order_by=["attributes.start_time DESC"],
        max_results=10_000,
    )
    selected = {}
    for run in candidates:
        condition = run.data.tags.get("atomic_run.id")
        run_study = (
            run.data.tags.get("experiment.id")
            or run.data.tags.get("experiment.ids", "").split(",")[0]
        )
        if (
            condition
            and run_study == study_id
            and condition not in selected
            and run.info.status == "FINISHED"
            and run.data.tags.get("result.durable_complete") == "true"
            and (
                schema_name is None
                or run.data.tags.get("result.schema.name") == schema_name
            )
            and (
                protocol_version is None
                or run.data.tags.get("protocol.version") == protocol_version
            )
            and (device is None or run.data.params.get("numerics/device") == device)
            and (
                timing_source is None
                or run.data.params.get("profiling/timing_source") == timing_source
            )
        ):
            selected[condition] = run
    return list(selected.values())


__all__ = ["latest_profile_runs"]
