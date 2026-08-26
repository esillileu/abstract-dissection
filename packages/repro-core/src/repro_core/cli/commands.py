"""Typed command services shared by domain CLI callbacks."""

from __future__ import annotations

import os
import re
from collections.abc import Callable

from repro_core.execution.definition import (
    ExecutionDefinition,
    RunOptions,
    RunOrder,
    RunSelection,
)
from repro_core.execution.parsing import parse_overrides
from repro_core.execution.planning import Planner
from repro_core.execution.runner import Runner, print_plans


def plan_command(
    domain: ExecutionDefinition,
    *,
    experiments: list[str],
    all_experiments: bool,
    atomic_runs: list[str],
    excluded_atomic_runs: list[str],
    seed_set: str | None,
    seeds: str | None,
    device: str | None,
    override_values: list[str],
    order: RunOrder,
) -> None:
    _validate_device(device)
    selection = RunSelection(
        tuple(experiments),
        all_experiments or not experiments,
        tuple(atomic_runs),
        tuple(excluded_atomic_runs),
        seeds,
        seed_set,
    )
    options = RunOptions(
        device=device, overrides=parse_overrides(override_values), order=order
    )
    print_plans(Planner(domain).build(selection, options))


def run_command(
    domain: ExecutionDefinition,
    *,
    experiments: list[str],
    all_experiments: bool,
    atomic_runs: list[str],
    excluded_atomic_runs: list[str],
    seed_set: str | None,
    seeds: str | None,
    device: str | None,
    override_values: list[str],
    order: RunOrder,
    dry_run: bool,
    progress: str,
    progress_every: int,
    tracking_uri: str | None,
    run_fn: Callable[..., object] | None = None,
) -> None:
    _validate_device(device)
    if progress not in {"auto", "none", "line", "tqdm"}:
        raise ValueError(f"unsupported progress mode: {progress}")
    if progress_every < 1:
        raise ValueError("--progress-every must be positive")
    if not all_experiments and not experiments:
        raise ValueError("run requires --all or --experiment/-e")
    selection = RunSelection(
        tuple(experiments),
        all_experiments,
        tuple(atomic_runs),
        tuple(excluded_atomic_runs),
        seeds,
        seed_set,
    )
    options = RunOptions(
        device=device,
        overrides=parse_overrides(override_values),
        progress=progress,
        progress_every=progress_every,
        order=order,
    )
    plans = Planner(domain).build(selection, options)
    print_plans(plans)
    if dry_run:
        return
    previous_uri = os.environ.get("REPRO_TRACKING_URI")
    previous_mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if tracking_uri is not None:
        os.environ["REPRO_TRACKING_URI"] = tracking_uri
        os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
    try:
        Runner(domain, run_fn=run_fn).run(plans, options)
    finally:
        if tracking_uri is not None:
            if previous_uri is None:
                os.environ.pop("REPRO_TRACKING_URI", None)
            else:
                os.environ["REPRO_TRACKING_URI"] = previous_uri
            if previous_mlflow_uri is None:
                os.environ.pop("MLFLOW_TRACKING_URI", None)
            else:
                os.environ["MLFLOW_TRACKING_URI"] = previous_mlflow_uri


def _validate_device(device: str | None) -> None:
    if device is not None and device != "cpu" and not re.fullmatch(r"cuda:\d+", device):
        raise ValueError(f"unsupported device: {device}")
