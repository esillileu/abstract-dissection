"""Run status inspection CLI commands."""

from __future__ import annotations

import json
from typing import Annotated

import typer

from dlfs.definition import DEFINITION
from dlfs.identity import Variant, Volume
from dlfs.tracking import resolve_tracking_uri
from repro_core.cli.types import (
    AtomicRuns,
    ExcludedAtomicRuns,
    Experiments,
    Overrides,
    cli_errors,
)
from repro_core.execution import RunOptions, RunSelection
from repro_core.execution.parsing import parse_overrides
from repro_core.execution.planning import Planner

from .common import _selected_variant


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

    from dlfs.execution.status import inspect_plan_status

    uri = resolve_tracking_uri(tracking_uri)
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
                )
                .to_executor_config()
                .get("protocol_version", "legacy")
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
