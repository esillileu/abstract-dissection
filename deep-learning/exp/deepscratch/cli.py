"""Domain-owned CLI callbacks for DeepScratch volumes and variants."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated

import typer

from exp.framework.cli.types import (
    AtomicRuns,
    ExcludedAtomicRuns,
    Experiments,
    Overrides,
    cli_errors,
)
from exp.framework.cli.commands import plan_command, run_command
from exp.framework.execution import RunOptions, RunOrder, RunSelection
from exp.framework.execution.parsing import parse_experiment_ids, parse_overrides
from exp.framework.execution.planning import Planner
from exp.framework.paths import StateCoordinate, StateOwner, WorkspacePaths

from .definition import DEFINITION
from .identity import Variant, Volume


def _selected_variant(variant: Variant, original: bool) -> Variant:
    if original and variant not in {Variant.IMPLEMENTED, Variant.ORIGINAL}:
        raise ValueError("-o cannot be combined with this --variant")
    return Variant.ORIGINAL if original else variant


def _writer_overrides(
    volume: Volume,
    variant: Variant,
    values: list[str],
) -> list[str]:
    schema = f"{volume.value}-{variant.value}"
    tags = {
        "domain.name": "deepscratch",
        "suite.name": volume.value,
        "deepscratch.volume": volume.value,
        "implementation.variant": variant.value,
        "experiment.id": "{experiment_id}",
        "condition.id": "{condition_id}",
        "result.schema.name": schema,
        "result.schema.version": "1",
    }
    return [
        *values,
        f"tracking.experiment=deepscratch.{volume.value}",
        f"tracking.tags={json.dumps(tags, separators=(',', ':'))}",
    ]


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
        experiments=experiment or [], all_experiments=all_experiments,
        atomic_runs=atomic_run or [], excluded_atomic_runs=exclude_atomic_run or [],
        seed_set=seed_set, seeds=seed, device=device,
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
        experiments=experiment or [], all_experiments=all_experiments,
        atomic_runs=atomic_run or [], excluded_atomic_runs=exclude_atomic_run or [],
        seed_set=seed_set, seeds=seed, device=device,
        override_values=_writer_overrides(volume, selected, override_values or []),
        order=order, dry_run=dry_run, progress=progress,
        progress_every=progress_every, tracking_uri=tracking_uri,
    )


@cli_errors
def check(
    volume: Annotated[Volume, typer.Argument()],
    experiment: Experiments = None,
    all_experiments: Annotated[bool, typer.Option("--all")] = False,
    atomic_run: AtomicRuns = None,
    exclude_atomic_run: ExcludedAtomicRuns = None,
    seed_set: Annotated[str | None, typer.Option("--seed-set")] = None,
    seed: Annotated[str | None, typer.Option("--seed")] = None,
    override_values: Overrides = None,
    tracking_uri: Annotated[str | None, typer.Option("--tracking-uri")] = None,
    variant: Annotated[Variant, typer.Option("--variant")] = Variant.IMPLEMENTED,
    original: Annotated[bool, typer.Option("-o")] = False,
    show: Annotated[
        str,
        typer.Option(
            "--show",
            help="Entries to print: incomplete, missing, or all.",
        ),
    ] = "incomplete",
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Show completed, running, failed, and missing planned runs."""
    if show not in {"incomplete", "missing", "all"}:
        raise ValueError("--show must be incomplete, missing, or all")
    selected = _selected_variant(variant, original)
    implementation = DEFINITION.implementation(volume, selected)
    overrides = parse_overrides(override_values or [])
    plans = Planner(implementation).build(
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
    from mlflow.tracking import MlflowClient

    from .execution.status import inspect_plan_status

    uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    report = inspect_plan_status(
        MlflowClient(tracking_uri=uri),
        plans,
        volume=volume,
        variant=selected,
        expected_protocols={
            (plan.experiment_id, plan.atomic_run_id): str(
                implementation.load_run_spec(
                    plan.path,
                    atomic_run_id=plan.atomic_run_id,
                    overrides=overrides,
                ).to_executor_config().get("protocol_version", "legacy")
            )
            for plan in plans
        },
    )
    if json_output:
        typer.echo(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
        return
    counts = report.counts
    typer.echo(
        f"deepscratch/{volume.value}/{selected.value}: "
        f"planned={len(report.entries)} completed={counts['completed']} "
        f"running={counts['running']} failed={counts['failed']} "
        f"missing={counts['missing']}"
    )
    for entry in report.entries:
        if show == "incomplete" and entry.status == "completed":
            continue
        if show == "missing" and entry.status != "missing":
            continue
        seed_label = "single" if entry.seed is None else str(entry.seed)
        detail = ""
        if entry.run_id is not None:
            detail = f" run={entry.run_id} source={entry.namespace}"
        typer.echo(
            f"{entry.status:9} {entry.experiment_id} "
            f"{entry.condition_id} seed={seed_label}{detail}"
        )


@cli_errors
def analyze(
    volume: Annotated[Volume, typer.Argument()],
    refresh_scope: Annotated[str | None, typer.Argument(hidden=True)] = None,
    experiment: Experiments = None,
    all_experiments: Annotated[bool, typer.Option("--all")] = False,
    tracking_uri: Annotated[str | None, typer.Option("--tracking-uri")] = None,
    output_dir: Annotated[Path | None, typer.Option("--output-dir")] = None,
    seed: Annotated[int | None, typer.Option("--seed")] = None,
    variant: Annotated[str, typer.Option("--variant")] = "implemented",
    original: Annotated[bool, typer.Option("-o")] = False,
    summary: Annotated[
        bool,
        typer.Option("--summary", "-s", help="Print per-condition scalar summaries."),
    ] = False,
    refresh: Annotated[
        bool,
        typer.Option(
            "--refresh",
            help="Refresh raw and analysis caches; append 'analysis' to refresh only analysis.",
        ),
    ] = False,
    run_id: Annotated[str | None, typer.Option("--run-id")] = None,
    error_style: Annotated[
        str,
        typer.Option(
            "--error-style",
            help="Seed variability display: band (shading) or errorbar.",
        ),
    ] = "band",
) -> None:
    if refresh_scope is not None and (not refresh or refresh_scope != "analysis"):
        raise ValueError("the optional --refresh scope must be 'analysis'")
    refresh_mode = "analysis" if refresh_scope == "analysis" else (
        "all" if refresh else None
    )
    if variant not in {"implemented", "original", "all"}:
        raise ValueError("--variant must be implemented, original, or all")
    if original and variant != "implemented":
        raise ValueError("-o cannot be combined with an explicit --variant")
    if original:
        variant = "original"
    if run_id is not None and variant == "all":
        raise ValueError("--run-id requires one explicit variant")
    if error_style not in {"band", "errorbar"}:
        raise ValueError("--error-style must be band or errorbar")
    variants = (
        (Variant.IMPLEMENTED, Variant.ORIGINAL)
        if variant == "all" else (Variant(variant),)
    )
    from .analysis.orchestrator import write_analysis
    from .analysis.paths import default_result_root, selection_directory

    selected_experiments = parse_experiment_ids(experiment or [])
    output_dir = selection_directory(
        output_dir
        or default_result_root(volume, selected_experiments, variants),
        volume=volume,
        study_ids=selected_experiments,
        variants=variants,
        seed=seed,
        run_id=run_id,
    )
    cache_dir = WorkspacePaths.from_environment(Path.cwd()).resolve(
        StateOwner.CACHE,
        StateCoordinate(
            "deepscratch",
            volume.value,
            selected_experiments[0] if len(selected_experiments) == 1 else "all",
            variant,
            "analysis",
        ),
    )
    if seed is not None:
        cache_dir /= f"seed-{seed}"
    elif run_id is not None:
        cache_dir /= f"run-{run_id[:8]}"
    typer.echo(
        f"selecting MLflow runs: deepscratch/{volume.value}/{variant}",
        err=True,
    )
    output = write_analysis(
        tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"),
        volume=volume,
        experiment_ids=selected_experiments,
        variants=variants,
        output_dir=output_dir,
        cache_dir=cache_dir,
        seed=seed,
        run_id=run_id,
        error_style=error_style,
        print_summary=summary,
        refresh=refresh_mode,
    )
    typer.echo(f"analysis: {output}")


@cli_errors
def profile(
    volume: Annotated[Volume, typer.Argument()],
    experiment: Experiments = None,
    variant: Annotated[Variant, typer.Option("--variant")] = Variant.IMPLEMENTED,
    device: Annotated[list[str] | None, typer.Option("--device")] = None,
    output_dir: Annotated[Path | None, typer.Option("--output-dir")] = None,
    condition: Annotated[list[str] | None, typer.Option("--condition")] = None,
    update_warmup: Annotated[int, typer.Option("--update-warmup")] = 20,
    update_repetitions: Annotated[int, typer.Option("--update-repetitions")] = 5,
    measured_updates: Annotated[int, typer.Option("--measured-updates")] = 50,
) -> None:
    if volume is not Volume.DS2:
        raise ValueError("DeepScratch DS1 has no declared profiles")
    if variant is not Variant.IMPLEMENTED:
        raise ValueError("selected profile supports only variant implemented")
    from .ds2.profile.cli import profile as ds2_profile

    selected_experiments = parse_experiment_ids(experiment or [])
    if len(selected_experiments) != 1:
        raise ValueError("profile requires exactly one experiment")
    selected_experiment = selected_experiments[0]
    if output_dir is None:
        from .ds2.profile.paths import profile_measurements

        output_dir = profile_measurements(selected_experiment)

    ds2_profile(
        experiment=experiment,
        device=device,
        output_dir=output_dir,
        condition=condition,
        update_warmup=update_warmup,
        update_repetitions=update_repetitions,
        measured_updates=measured_updates,
    )
    typer.echo(f"profile cache: {output_dir}")
