"""Check command implementation comparing declared plans with recorded MLflow state."""

from __future__ import annotations

import typer
from mlflow.tracking import MlflowClient

from repro_core.execution import RunOptions, RunSelection
from repro_core.execution.parsing import parse_overrides
from repro_core.execution.planning import Planner

from ..execution.definitions import DEFINITION
from ..execution.identity import ConditionIdentity
from ..storage.publication import PublicationService
from ..tracking import resolve_tracking_uri


def check_command_impl(
    suite: str,
    experiment: list[str] | None = None,
    all_experiments: bool = False,
    atomic_run: list[str] | None = None,
    exclude_atomic_run: list[str] | None = None,
    seed_set: str | None = None,
    seed: str | None = None,
    override_values: list[str] | None = None,
    tracking_uri: str | None = None,
) -> None:
    """Compare declared plans with recorded F2 run state in MLflow."""
    suite_def = DEFINITION.get_suite(suite)
    overrides = parse_overrides(override_values or [])
    plans = Planner(suite_def).build(
        RunSelection(
            experiment_ids=tuple(experiment or []),
            all_experiments=all_experiments or not experiment,
            atomic_run_ids=tuple(atomic_run or []),
            excluded_atomic_run_ids=tuple(exclude_atomic_run or []),
            seed_values=seed,
            seed_set=seed_set,
        ),
        RunOptions(overrides=overrides),
    )
    uri = resolve_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=uri)
    experiment_name = f"f2.{suite}"
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        typer.echo(
            f"No MLflow experiment found for '{experiment_name}'. "
            f"All {len(plans)} planned runs are missing."
        )
        return

    publication = PublicationService(tracking_uri=uri)
    all_finished_runs = publication.get_finished_runs_for_experiment(experiment_name)
    completed_count = 0
    missing_count = 0
    for p in plans:
        spec = suite_def.load_run_spec(
            p.path, atomic_run_id=p.atomic_run_id, overrides=overrides
        )
        seeded_spec = spec.with_seed(p.seed) if p.seed is not None else spec
        config = seeded_spec.to_executor_config()
        condition = ConditionIdentity.from_config(config, suite_id=suite)
        finished_run = publication.find_matching_finished_run(
            experiment_name,
            condition,
            atomic_run_id=p.atomic_run_id,
            cached_runs=all_finished_runs,
        )
        if finished_run is not None:
            completed_count += 1
        else:
            missing_count += 1
    typer.echo(
        f"f2/{suite}: planned={len(plans)} completed={completed_count} missing={missing_count}"
    )


__all__ = ["check_command_impl"]
