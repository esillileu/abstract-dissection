"""Cutover gate for immutable historical writer namespaces."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient

from .identity import LEGACY_NAMESPACES


@dataclass(frozen=True)
class ActiveLegacyRun:
    namespace: str
    run_id: str
    run_key: str | None
    experiment_id: str | None
    condition_id: str | None
    seed: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def find_active_legacy_runs(client: MlflowClient) -> list[ActiveLegacyRun]:
    active: list[ActiveLegacyRun] = []
    for namespace in dict.fromkeys(LEGACY_NAMESPACES.values()):
        experiment = client.get_experiment_by_name(namespace)
        if experiment is None:
            continue
        runs = client.search_runs(
            [experiment.experiment_id],
            filter_string="attributes.status = 'RUNNING'",
            run_view_type=ViewType.ALL,
            max_results=10_000,
        )
        for run in runs:
            tags = run.data.tags
            active.append(ActiveLegacyRun(
                namespace=namespace,
                run_id=run.info.run_id,
                run_key=tags.get("run.key"),
                experiment_id=tags.get("experiment.id") or tags.get("experiment.ids"),
                condition_id=tags.get("condition.id") or tags.get("atomic_run.id"),
                seed=tags.get("seed") or tags.get("master_seed"),
            ))
    return active


def require_cutover_safe(client: MlflowClient) -> None:
    active = find_active_legacy_runs(client)
    if active:
        identities = ", ".join(f"{run.namespace}/{run.run_id}" for run in active)
        raise RuntimeError(f"legacy cutover blocked by RUNNING runs: {identities}")
