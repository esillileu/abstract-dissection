"""Execution planning and run CLI commands."""

from __future__ import annotations

from typing import Annotated

import typer

from dlfs.definition import DEFINITION
from dlfs.identity import Variant, Volume
from dlfs.tracking import resolve_tracking_uri
from repro_core.cli.commands import plan_command, run_command
from repro_core.cli.types import (
    AtomicRuns,
    ExcludedAtomicRuns,
    Experiments,
    Overrides,
    cli_errors,
)
from repro_core.execution import RunOrder
from repro_mlflow import run_yaml

from .common import _selected_variant, _writer_overrides


@cli_errors
def plan(
    volume: Annotated[Volume, typer.Argument()],
    experiment: Experiments = None,
    all_experiments: Annotated[bool, typer.Option("--all")] = False,
    atomic_run: AtomicRuns = None,
    exclude_atomic_run: ExcludedAtomicRuns = None,
    seed_set: Annotated[str | None, typer.Option("--seed-set")] = None,
    seed: Annotated[str | None, typer.Option("--seed")] = None,
    device: Annotated[str | None, typer.Option("--device")] = None,
    override_values: Overrides = None,
    order: Annotated[RunOrder, typer.Option("--order")] = RunOrder.CATALOG_FIRST,
    variant: Annotated[Variant, typer.Option("--variant")] = Variant.IMPLEMENTED,
    original: Annotated[bool, typer.Option("-o")] = False,
) -> None:
    selected = _selected_variant(variant, original)
    plan_command(
        DEFINITION.implementation(volume, selected),
        experiments=experiment or [],
        all_experiments=all_experiments,
        atomic_runs=atomic_run or [],
        excluded_atomic_runs=exclude_atomic_run or [],
        seed_set=seed_set,
        seeds=seed,
        device=device,
        override_values=_writer_overrides(volume, selected, override_values or []),
        order=order,
    )


@cli_errors
def run(
    volume: Annotated[Volume, typer.Argument()],
    experiment: Experiments = None,
    all_experiments: Annotated[bool, typer.Option("--all")] = False,
    atomic_run: AtomicRuns = None,
    exclude_atomic_run: ExcludedAtomicRuns = None,
    seed_set: Annotated[str | None, typer.Option("--seed-set")] = None,
    seed: Annotated[str | None, typer.Option("--seed")] = None,
    device: Annotated[str | None, typer.Option("--device")] = None,
    override_values: Overrides = None,
    order: Annotated[RunOrder, typer.Option("--order")] = RunOrder.CATALOG_FIRST,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
    progress: Annotated[str, typer.Option("--progress")] = "auto",
    progress_every: Annotated[int, typer.Option("--progress-every")] = 10,
    tracking_uri: Annotated[str | None, typer.Option("--tracking-uri")] = None,
    variant: Annotated[Variant, typer.Option("--variant")] = Variant.IMPLEMENTED,
    original: Annotated[bool, typer.Option("-o")] = False,
) -> None:
    selected = _selected_variant(variant, original)
    run_command(
        DEFINITION.implementation(volume, selected),
        experiments=experiment or [],
        all_experiments=all_experiments,
        atomic_runs=atomic_run or [],
        excluded_atomic_runs=exclude_atomic_run or [],
        seed_set=seed_set,
        seeds=seed,
        device=device,
        override_values=_writer_overrides(volume, selected, override_values or []),
        order=order,
        dry_run=dry_run,
        progress=progress,
        progress_every=progress_every,
        tracking_uri=None if dry_run else resolve_tracking_uri(tracking_uri),
        run_fn=run_yaml,
    )
