"""Typer callbacks and registration contract for the DS1 domain."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from exp.cli_support import (
    AtomicRuns,
    ExcludedAtomicRuns,
    Experiments,
    Overrides,
    cli_errors,
)
from exp.domain import DomainDefinition, RunOrder


DEFINITION = DomainDefinition(
    name="ds1",
    config_root=Path("exp/ds1/config"),
    spec_module="exp.ds1.spec",
    executor_module="exp.ds1.executor",
    analysis_module="exp.ds1.analyze.render",
    original_run_module="exp.ds1.original.run.api",
    original_render_module="exp.ds1.original.render.api",
    original_summary_module="exp.ds1.original.summary",
    original_experiments=(
        "e01", "e02", "e03", "e04", "e05", "e06", "e07", "e09", "e10"
    ),
)


@cli_errors
def plan(
    experiment: Experiments = None,
    all_experiments: Annotated[bool, typer.Option("--all")] = False,
    atomic_run: AtomicRuns = None,
    exclude_atomic_run: ExcludedAtomicRuns = None,
    seed_set: Annotated[str | None, typer.Option("--seed-set")] = None,
    seed: Annotated[str | None, typer.Option("--seed")] = None,
    device: Annotated[str | None, typer.Option("--device")] = None,
    override_values: Overrides = None,
    order: Annotated[RunOrder, typer.Option("--order")] = RunOrder.CATALOG_FIRST,
) -> None:
    from exp.commands import plan_command

    plan_command(
        DEFINITION,
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


@cli_errors
def run(
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
    original: Annotated[bool, typer.Option("-o", "--original")] = False,
    force: Annotated[bool, typer.Option("--force")] = False,
    output_dir: Annotated[Path | None, typer.Option("--output-dir")] = None,
) -> None:
    from exp.commands import run_command

    run_command(
        DEFINITION,
        experiments=experiment or [],
        all_experiments=all_experiments,
        atomic_runs=atomic_run or [],
        excluded_atomic_runs=exclude_atomic_run or [],
        seed_set=seed_set,
        seeds=seed,
        device=device,
        override_values=override_values or [],
        order=order,
        dry_run=dry_run,
        progress=progress,
        progress_every=progress_every,
        tracking_uri=tracking_uri,
        original=original,
        force=force,
        output_dir=output_dir,
    )


@cli_errors
def analyze(
    experiment: Experiments = None,
    all_experiments: Annotated[bool, typer.Option("--all")] = False,
    tracking_uri: Annotated[str | None, typer.Option("--tracking-uri")] = None,
    error_style: Annotated[str, typer.Option("--error-style")] = "band",
    output_dir: Annotated[Path | None, typer.Option("--output-dir")] = None,
    seed: Annotated[int | None, typer.Option("--seed")] = None,
    summary: Annotated[bool, typer.Option("-s", "--summary")] = False,
    original: Annotated[bool, typer.Option("-o", "--original")] = False,
) -> None:
    from exp.commands import analyze_command

    analyze_command(
        DEFINITION,
        experiments=experiment or [],
        all_experiments=all_experiments,
        tracking_uri=tracking_uri,
        error_style=error_style,
        output_dir=output_dir,
        seed=seed,
        summary=summary,
        original=original,
    )
