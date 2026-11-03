"""Plan command implementation for F2 reproduction suites."""

from __future__ import annotations

from typing import Annotated

import typer

from repro_core.execution.definition import RunOrder

from ..execution.definitions import DEFINITION

Experiments = Annotated[
    list[str] | None,
    typer.Option("-e", "--experiment", help="Experiment IDs or ranges."),
]
AtomicRuns = Annotated[
    list[str] | None,
    typer.Option("-a", "--atomic-run", help="Include atomic run IDs."),
]
ExcludedAtomicRuns = Annotated[
    list[str] | None,
    typer.Option("-x", "--exclude-atomic-run", help="Exclude atomic run IDs."),
]
Overrides = Annotated[
    list[str] | None,
    typer.Option("--set", metavar="KEY=VALUE", help="Override YAML values."),
]


def plan_command_impl(
    suite: str,
    experiment: list[str] | None = None,
    all_experiments: bool = False,
    atomic_run: list[str] | None = None,
    exclude_atomic_run: list[str] | None = None,
    seed_set: str | None = None,
    seed: str | None = None,
    device: str | None = None,
    override_values: list[str] | None = None,
    order: RunOrder = RunOrder.CATALOG_FIRST,
) -> None:
    """Inspect expanded experiment run plans for an F2 suite."""
    if suite == "corpus":
        typer.echo("For corpus pipeline planning, use: repro f2 corpus plan --help")
        return
    from repro_core.cli.commands import plan_command

    suite_def = DEFINITION.get_suite(suite)
    plan_command(
        suite_def,
        experiments=experiment or [],
        all_experiments=all_experiments,
        atomic_runs=atomic_run or [],
        excluded_atomic_runs=exclude_atomic_run or [],
        seed_set=seed_set,
        seeds=seed,
        device=device,
        override_values=override_values or [],
        order=order,
    )


__all__ = [
    "AtomicRuns",
    "ExcludedAtomicRuns",
    "Experiments",
    "Overrides",
    "plan_command_impl",
]
