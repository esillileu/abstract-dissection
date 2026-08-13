"""Domain-owned CLI callbacks for DeepScratch volumes and variants."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated

import typer

from exp.analyze import analysis_scope
from exp.cli_support import AtomicRuns, ExcludedAtomicRuns, Experiments, Overrides, cli_errors
from exp.commands import analyze_command, plan_command, run_command
from exp.domain import RunOrder
from exp.domain import RunOptions, RunSelection
from exp.parsing import parse_overrides
from exp.parsing import parse_experiment_ids
from exp.planning import Planner
from exp.framework.state import StateCoordinate, StateOwner, WorkspacePaths

from .definition import DEFINITION
from .identity import Variant, Volume
from .identity import legacy_namespace


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

    from .status import inspect_plan_status

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
    experiment: Experiments = None,
    all_experiments: Annotated[bool, typer.Option("--all")] = False,
    tracking_uri: Annotated[str | None, typer.Option("--tracking-uri")] = None,
    error_style: Annotated[str, typer.Option("--error-style")] = "band",
    output_dir: Annotated[Path | None, typer.Option("--output-dir")] = None,
    seed: Annotated[int | None, typer.Option("--seed")] = None,
    summary: Annotated[bool, typer.Option("-s", "--summary")] = False,
    variant: Annotated[str, typer.Option("--variant")] = "implemented",
    original: Annotated[bool, typer.Option("-o")] = False,
    run_id: Annotated[str | None, typer.Option("--run-id")] = None,
) -> None:
    if variant not in {"implemented", "original", "all"}:
        raise ValueError("--variant must be implemented, original, or all")
    if original and variant != "implemented":
        raise ValueError("-o cannot be combined with an explicit --variant")
    if original:
        variant = "original"
    if run_id is not None and variant == "all":
        raise ValueError("--run-id requires one explicit variant")
    variants = (
        (Variant.IMPLEMENTED, Variant.ORIGINAL)
        if variant == "all" else (Variant(variant),)
    )
    for selected in variants:
        implementation = DEFINITION.implementation(volume, selected)
        if output_dir is None:
            selected_experiments = parse_experiment_ids(experiment or [])
            experiment_part = (
                selected_experiments[0]
                if len(selected_experiments) == 1 and not all_experiments
                else "all"
            )
            scoped_output = WorkspacePaths.from_environment(Path.cwd()).resolve(
                StateOwner.ARTIFACT,
                StateCoordinate(
                    "deepscratch",
                    volume.value,
                    experiment_part,
                    selected.value,
                    "analysis",
                ),
            )
        else:
            scoped_output = output_dir
        if output_dir is not None and len(variants) > 1:
            scoped_output = output_dir / selected.value
        with analysis_scope(
            experiment_aliases={
                implementation.name: (
                    f"deepscratch.{volume.value}",
                    legacy_namespace(volume, selected),
                )
            },
            variant=selected.value,
            run_ids=() if run_id is None else (run_id,),
        ):
            analyze_command(
                implementation,
                experiments=experiment or [], all_experiments=all_experiments,
                tracking_uri=tracking_uri, error_style=error_style,
                output_dir=scoped_output, seed=seed, summary=summary,
            )
    if variant == "all":
        from .cross_variant import write_comparison_table

        selected_experiments = parse_experiment_ids(experiment or [])
        comparison_root = WorkspacePaths.from_environment(Path.cwd()).resolve(
            StateOwner.ARTIFACT,
            StateCoordinate(
                "deepscratch",
                volume.value,
                selected_experiments[0] if len(selected_experiments) == 1 else "all",
                "comparison",
            ),
        )
        output = write_comparison_table(
            tracking_uri
            or os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"),
            volume=volume,
            experiment_ids=selected_experiments,
            output_dir=comparison_root,
        )
        typer.echo(f"comparison: {output}")


@cli_errors
def profile(
    volume: Annotated[Volume, typer.Argument()],
    experiment: Experiments = None,
    variant: Annotated[Variant, typer.Option("--variant")] = Variant.IMPLEMENTED,
    device: Annotated[list[str] | None, typer.Option("--device")] = None,
    mode: Annotated[str, typer.Option("--mode")] = "all",
    output_dir: Annotated[Path | None, typer.Option("--output-dir")] = None,
    vsweap: Annotated[bool, typer.Option("--vsweap")] = False,
    vocab_size: Annotated[list[int] | None, typer.Option("--vocab-size")] = None,
    condition: Annotated[list[str] | None, typer.Option("--condition")] = None,
    update_warmup: Annotated[int, typer.Option("--update-warmup")] = 20,
    update_repetitions: Annotated[int, typer.Option("--update-repetitions")] = 5,
    measured_updates: Annotated[int, typer.Option("--measured-updates")] = 50,
    vsweap_timing: Annotated[str, typer.Option("--vsweap-timing")] = "window",
    reverse_vocab_order: Annotated[bool, typer.Option("--reverse-vocab-order")] = False,
    tracking_uri: Annotated[str | None, typer.Option("--tracking-uri")] = None,
    record: Annotated[bool, typer.Option("--record/--no-record")] = True,
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
        output_dir = WorkspacePaths.from_environment(Path.cwd()).resolve(
            StateOwner.CACHE,
            StateCoordinate(
                "deepscratch",
                volume.value,
                selected_experiment,
                variant.value,
                "profile",
            ),
        )

    ds2_profile(
        experiment=experiment,
        device=device,
        mode=mode,
        output_dir=output_dir,
        vsweap=vsweap,
        vocab_size=vocab_size,
        condition=condition,
        update_warmup=update_warmup,
        update_repetitions=update_repetitions,
        measured_updates=measured_updates,
        vsweap_timing=vsweap_timing,
        reverse_vocab_order=reverse_vocab_order,
    )
    if record:
        from .profile_results import record_profile_result

        run_id = record_profile_result(
            tracking_uri
            or os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"),
            volume=volume.value,
            experiment_id=selected_experiment,
            variant=variant.value,
            output=output_dir,
        )
        typer.echo(f"profile run: {run_id}")


@cli_errors
def import_legacy(
    domain: Annotated[str, typer.Argument()],
    volume: Annotated[Volume, typer.Argument()],
    variant: Annotated[Variant, typer.Option("--variant")],
    input_path: Annotated[Path, typer.Option("--input", "-i")],
    tracking_uri: Annotated[str | None, typer.Option("--tracking-uri")] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
) -> None:
    from .legacy_import import import_legacy_archive

    if domain != "deepscratch":
        raise ValueError("import-legacy supports only the deepscratch domain")
    uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    report = import_legacy_archive(
        uri, input_path, volume=volume, variant=variant, dry_run=dry_run
    )
    typer.echo(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
