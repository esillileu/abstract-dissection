"""Experiment analysis and visualization CLI commands."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from dlfs.identity import Variant, Volume
from dlfs.tracking import resolve_tracking_uri
from repro_core.cli.types import Experiments, cli_errors
from repro_core.context.paths import StateCoordinate, StateOwner, WorkspacePaths
from repro_core.execution.parsing import parse_experiment_ids


@cli_errors
def analyze(
    volume: Annotated[Volume, typer.Argument()],
    refresh_scope: Annotated[str | None, typer.Argument(hidden=True)] = None,
    experiment: Experiments = None,
    all_experiments: Annotated[bool, typer.Option("--all")] = False,
    tracking_uri: Annotated[str | None, typer.Option("--tracking-uri")] = None,
    output_dir: Annotated[Path | None, typer.Option("--output-dir")] = None,
    seed: Annotated[int | None, typer.Option("--seed")] = None,
    device: Annotated[str | None, typer.Option("--device")] = None,
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
    refresh_mode = (
        "analysis" if refresh_scope == "analysis" else ("all" if refresh else None)
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
    if device is not None and device != "cpu" and not device.startswith("cuda:"):
        raise ValueError("--device must be cpu or cuda:<index>")
    variants = (
        (Variant.IMPLEMENTED, Variant.ORIGINAL)
        if variant == "all"
        else (Variant(variant),)
    )
    from dlfs.analysis.orchestrator import write_analysis
    from dlfs.analysis.paths import default_result_root, selection_directory

    selected_experiments = parse_experiment_ids(experiment or [])
    output_dir = selection_directory(
        output_dir or default_result_root(volume, selected_experiments, variants),
        volume=volume,
        study_ids=selected_experiments,
        variants=variants,
        seed=seed,
        run_id=run_id,
    )
    if device is not None:
        output_dir /= device.replace(":", "-")
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
    if device is not None:
        cache_dir /= f"device-{device.replace(':', '-')}"
    typer.echo(
        f"selecting MLflow runs: deepscratch/{volume.value}/{variant}",
        err=True,
    )
    output = write_analysis(
        resolve_tracking_uri(tracking_uri),
        volume=volume,
        experiment_ids=selected_experiments,
        variants=variants,
        output_dir=output_dir,
        cache_dir=cache_dir,
        seed=seed,
        device=device,
        run_id=run_id,
        error_style=error_style,
        print_summary=summary,
        refresh=refresh_mode,
    )
    typer.echo(f"analysis: {output}")
