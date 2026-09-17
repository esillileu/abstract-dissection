"""Profiling CLI commands."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from dlfs.identity import Variant, Volume
from dlfs.tracking import resolve_tracking_uri
from repro_core.cli.types import Experiments, cli_errors
from repro_core.execution.parsing import parse_experiment_ids


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
    tracking_uri: Annotated[str | None, typer.Option("--tracking-uri")] = None,
) -> None:
    resolve_tracking_uri(tracking_uri)
    if volume is not Volume.DS2:
        raise ValueError("DeepScratch DS1 has no declared profiles")
    if variant is not Variant.IMPLEMENTED:
        raise ValueError("selected profile supports only variant implemented")
    from dlfs.ds2.profile.cli import profile as ds2_profile

    selected_experiments = parse_experiment_ids(experiment or [])
    if len(selected_experiments) != 1:
        raise ValueError("profile requires exactly one experiment")
    selected_experiment = selected_experiments[0]
    if output_dir is None:
        from dlfs.ds2.profile.paths import profile_measurements

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
