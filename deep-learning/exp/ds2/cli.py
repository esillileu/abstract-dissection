"""Typer callbacks and registration contract for the DS2 domain."""

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
    name="ds2",
    config_root=Path("exp/ds2/config"),
    spec_module="exp.ds2.spec",
    executor_module="exp.ds2.executor",
    analysis_module="exp.ds2.analyze.render",
    original_run_module="exp.ds2.original.run.api",
    original_render_module="exp.ds2.original.render.api",
    original_summary_module="exp.ds2.original.summary",
    original_experiments=("e01", "e02", "e03", "e04", "e06", "e07", "e08"),
    original_default_order=("e01", "e03", "e04", "e06", "e07", "e08", "e02"),
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


@cli_errors
def profile(
    experiment: Experiments = None,
    device: Annotated[
        list[str] | None,
        typer.Option("--device", help="Repeat to select CPU/CUDA devices."),
    ] = None,
    condition: Annotated[
        list[str] | None,
        typer.Option("--condition", help="Repeat to select e02 model conditions."),
    ] = None,
    mode: Annotated[
        str,
        typer.Option("--mode", help="all, update, or modules."),
    ] = "all",
    component: Annotated[
        list[str] | None,
        typer.Option("--component", help="Repeat to select module sections."),
    ] = None,
    batch_size: Annotated[int, typer.Option("--batch-size")] = 100,
    epochs: Annotated[int, typer.Option("--epochs")] = 10,
    update_warmup: Annotated[int, typer.Option("--update-warmup")] = 5,
    update_repetitions: Annotated[
        int,
        typer.Option("--update-repetitions"),
    ] = 20,
    measured_updates: Annotated[int, typer.Option("--measured-updates")] = 1,
    module_warmup: Annotated[int, typer.Option("--module-warmup")] = 5,
    module_iterations: Annotated[
        int,
        typer.Option("--module-iterations"),
    ] = 20,
    output_dir: Annotated[Path | None, typer.Option("--output-dir")] = None,
) -> None:
    from exp.ds2.profile.e02.api import DEFAULT_RESULTS, run
    from exp.parsing import parse_experiment_ids

    selected = parse_experiment_ids(experiment or [])
    if selected != ["e02"]:
        raise ValueError("DS2 profiling currently requires exactly -e 02")
    if (
        min(
            batch_size,
            epochs,
            update_repetitions,
            measured_updates,
            module_iterations,
        )
        < 1
    ):
        raise ValueError(
            "batch size, epochs, repetitions, and measured iterations must be positive"
        )
    if min(update_warmup, module_warmup) < 0:
        raise ValueError("warmup counts must be non-negative")

    run(
        devices=tuple(device or ("cpu", "cuda:0")),
        conditions=tuple(condition) if condition else None,
        mode=mode,
        components=(
            tuple(value.replace("-", "_") for value in component) if component else None
        ),
        batch_size=batch_size,
        epochs=epochs,
        update_warmup=update_warmup,
        update_repetitions=update_repetitions,
        measured_updates=measured_updates,
        module_warmup=module_warmup,
        module_iterations=module_iterations,
        output_dir=output_dir or DEFAULT_RESULTS,
    )
