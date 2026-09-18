"""MLflow client delegation and seed-aware run selection."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from repro_core.context.paths import WorkspacePaths


class AnalysisClient:
    """Delegate MLflow calls while carrying analysis-only selection state."""

    def __init__(
        self,
        client,
        *,
        seed: int | None = None,
    ) -> None:
        self._client = client
        self.analysis_seed = None if seed is None else str(seed)
        self.analysis_selections: list[dict[str, object]] = []

    def __getattr__(self, name):
        return getattr(self._client, name)

    def record_analysis_selection(self, selection: dict[str, object]) -> None:
        self.analysis_selections.append(selection)


@dataclass(frozen=True)
class RunRef:
    run_id: str
    atomic_run_id: str
    seed: str
    start_time: int
    local_artifact_root: Path | None = None


def completed_seed_runs(
    client,
    *,
    experiment_name: str,
    group_id: str,
    atomic_run_ids: Iterable[str],
    protocol_version: str | None = None,
) -> dict[str, list[RunRef]]:
    """Return the newest completed attempt for every (condition, seed)."""
    wanted = tuple(atomic_run_ids)
    grouped: dict[str, list[RunRef]] = {atomic: [] for atomic in wanted}
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return grouped
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=(
            "attributes.status = 'FINISHED' and "
            f"tags.`execution_group.id` = '{group_id}'"
        ),
        order_by=["attributes.start_time DESC"],
        max_results=5_000,
    )
    selected: dict[tuple[str, str], RunRef] = {}
    for run in runs:
        tags = run.data.tags
        if tags.get("run.type") != "seed_trial":
            continue
        if tags.get("execution_group.id") != group_id:
            continue
        if (
            protocol_version is not None
            and tags.get("protocol.version", "legacy") != protocol_version
        ):
            continue
        atomic = tags.get("atomic_run.id", "")
        if atomic not in grouped:
            continue
        seed = run.data.params.get(
            "seed/master", run.data.params.get("seed", run.info.run_id)
        )
        selected_seed = getattr(client, "analysis_seed", None)
        if selected_seed is not None and str(seed) != selected_seed:
            continue
        key = (atomic, str(seed))
        selected.setdefault(
            key,
            RunRef(
                run.info.run_id,
                atomic,
                str(seed),
                int(run.info.start_time or 0),
                _local_artifact_root(tags.get("run.key"), tags),
            ),
        )
    for (atomic, _), run in selected.items():
        grouped[atomic].append(run)
    for runs_for_condition in grouped.values():
        runs_for_condition.sort(key=lambda run: run.seed)
    record_selection = getattr(client, "record_analysis_selection", None)
    if record_selection is not None:
        record_selection(
            {
                "experiment_name": experiment_name,
                "group_id": group_id,
                "atomic_run_ids": list(wanted),
                "protocol_version": protocol_version,
                "run_ids": sorted(
                    run.run_id
                    for runs_for_condition in grouped.values()
                    for run in runs_for_condition
                ),
            }
        )
    return grouped


def _local_artifact_root(
    run_key: str | None,
    tags=None,
) -> Path | None:
    if not run_key:
        return None
    metadata = tags or {}
    domain = metadata.get("domain.name")
    suite = metadata.get("suite.name")
    study = (
        metadata.get("experiment.id")
        or metadata.get("experiment.ids", "").split(",")[0]
    )
    variant = metadata.get("implementation.variant")
    if all((domain, suite, study, variant)):
        staging = (
            WorkspacePaths.from_environment(Path.cwd()).run_staging(
                domain=str(domain),
                suite=str(suite),
                study=str(study),
                variant=str(variant),
                run_key=run_key,
            )
            / "record"
        )
        if staging.is_dir():
            return staging
    return None
