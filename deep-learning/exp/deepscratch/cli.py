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
    return [
        *values,
        f"tracking.experiment=deepscratch.{volume.value}",
        "tracking.tags.domain.name=deepscratch",
        f"tracking.tags.deepscratch.volume={volume.value}",
        f"tracking.tags.implementation.variant={variant.value}",
        "tracking.tags.experiment.id={experiment_id}",
        "tracking.tags.condition.id={condition_id}",
        f"tracking.tags.result.schema.name={schema}",
        "tracking.tags.result.schema.version=1",
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
        original=False, force=False, output_dir=None,
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
) -> None:
    if variant not in {"implemented", "original", "all"}:
        raise ValueError("--variant must be implemented, original, or all")
    variants = (
        (Variant.IMPLEMENTED, Variant.ORIGINAL)
        if variant == "all" else (Variant(variant),)
    )
    for selected in variants:
        implementation = DEFINITION.implementation(volume, selected)
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
        ):
            analyze_command(
                implementation,
                experiments=experiment or [], all_experiments=all_experiments,
                tracking_uri=tracking_uri, error_style=error_style,
                output_dir=scoped_output, seed=seed, summary=summary,
                original=False,
            )


@cli_errors
def profile(
    volume: Annotated[Volume, typer.Argument()],
    experiment: Experiments = None,
    variant: Annotated[Variant, typer.Option("--variant")] = Variant.IMPLEMENTED,
    device: Annotated[list[str] | None, typer.Option("--device")] = None,
    mode: Annotated[str, typer.Option("--mode")] = "all",
    output_dir: Annotated[Path | None, typer.Option("--output-dir")] = None,
) -> None:
    if volume is not Volume.DS2:
        raise ValueError("DeepScratch DS1 has no declared profiles")
    if variant is not Variant.IMPLEMENTED:
        raise ValueError("selected profile supports only variant implemented")
    from exp.ds2.cli import profile as ds2_profile

    ds2_profile(
        experiment=experiment,
        device=device,
        mode=mode,
        output_dir=output_dir,
    )


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
