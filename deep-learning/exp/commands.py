"""Typed command services shared by the domain CLI callbacks."""

from __future__ import annotations

import importlib
import os
from pathlib import Path

from exp.domain import DomainDefinition, RunOptions, RunOrder, RunSelection
from exp.parsing import parse_experiment_ids, parse_overrides
from exp.planning import Planner
from exp.runner import Runner, print_plans


def plan_command(
    domain: DomainDefinition,
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
        device=device,
        overrides=parse_overrides(override_values),
        order=order,
    )
    print_plans(Planner(domain).build(selection, options))


def run_command(
    domain: DomainDefinition,
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
    original: bool,
    force: bool,
    output_dir: Path | None,
) -> None:
    _validate_device(device)
    if progress not in {"auto", "none", "line", "tqdm"}:
        raise ValueError(f"unsupported progress mode: {progress}")
    if progress_every < 1:
        raise ValueError("--progress-every must be positive")
    if original:
        conflicts = [
            (atomic_runs or excluded_atomic_runs, "atomic-run selection"),
            (seed_set is not None, "--seed-set"),
            (seeds is not None, "--seed"),
            (device is not None, "--device"),
            (bool(override_values), "YAML overrides"),
            (tracking_uri is not None, "--tracking-uri"),
        ]
        for active, label in conflicts:
            if active:
                raise ValueError(f"{label} is incompatible with --original")
        selected = select_original_experiments(domain, experiments)
        if all_experiments and experiments:
            raise ValueError("choose at most one of --all or --experiment/-e")
        if dry_run:
            print(f"{domain.name}: original seed=1: {', '.join(selected)}")
            return
        run_original(
            domain,
            selected,
            force=force,
            output_dir=output_dir,
        )
        return
    if force:
        raise ValueError("--force requires --original")
    if output_dir is not None:
        raise ValueError("--output-dir requires --original")
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
    previous_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if tracking_uri is not None:
        os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
    try:
        Runner(domain).run(plans, options)
    finally:
        if tracking_uri is not None:
            if previous_uri is None:
                os.environ.pop("MLFLOW_TRACKING_URI", None)
            else:
                os.environ["MLFLOW_TRACKING_URI"] = previous_uri


def analyze_command(
    domain: DomainDefinition,
    *,
    experiments: list[str],
    all_experiments: bool,
    tracking_uri: str | None,
    error_style: str,
    output_dir: Path | None,
    seed: int | None,
    summary: bool,
    original: bool,
) -> None:
    if error_style not in {"band", "errorbar"}:
        raise ValueError(f"unsupported error style: {error_style}")
    if all_experiments and experiments:
        raise ValueError("choose at most one of --all or --experiment/-e")
    if original:
        if seed is not None:
            raise ValueError("--seed is incompatible with --original")
        if tracking_uri is not None:
            raise ValueError("--tracking-uri is incompatible with --original")
        selected = (
            select_original_summary_experiments(domain, experiments)
            if summary
            else select_original_experiments(domain, experiments)
        )
        if summary:
            summarize_original(domain, selected, output_dir=output_dir)
        else:
            analyze_original(domain, selected, output_dir=output_dir)
        return
    module = importlib.import_module(domain.analysis_module)
    module.analyze(
        experiments=[] if all_experiments else experiments,
        tracking_uri=tracking_uri,
        error_style=error_style,
        output_dir=output_dir,
        seed=seed,
        summary=summary,
    )


def select_original_experiments(
    domain: DomainDefinition, requested: list[str]
) -> list[str]:
    available = domain.original_experiments
    if not requested:
        return list(domain.original_order())
    selected = parse_experiment_ids(requested)
    unknown = [item for item in selected if item not in available]
    if unknown:
        raise ValueError(
            f"experiments have no registered original trials for {domain.name}: "
            + ", ".join(unknown)
        )
    return selected


def select_original_summary_experiments(
    domain: DomainDefinition, requested: list[str]
) -> list[str]:
    if domain.original_summary_module is None:
        raise ValueError(f"{domain.name} has no original summaries")
    module = importlib.import_module(domain.original_summary_module)
    available = tuple(module.SUPPORTED_EXPERIMENTS)
    if not requested:
        return list(available)
    selected = parse_experiment_ids(requested)
    unknown = [item for item in selected if item not in available]
    if unknown:
        raise ValueError(
            f"experiments have no original summary for {domain.name}: "
            + ", ".join(unknown)
        )
    return selected


def run_original(
    domain: DomainDefinition,
    experiments: list[str],
    *,
    force: bool,
    output_dir: Path | None,
) -> None:
    if domain.original_run_module is None or domain.original_render_module is None:
        raise ValueError(f"{domain.name} does not support original trials")
    root = output_dir or domain.config_root.parent / "results" / "original"
    importlib.import_module(domain.original_run_module).run(
        experiments, root=root, force=force
    )
    importlib.import_module(domain.original_render_module).render(
        experiments, root=root
    )


def analyze_original(
    domain: DomainDefinition,
    experiments: list[str],
    *,
    output_dir: Path | None,
) -> None:
    if domain.original_render_module is None:
        raise ValueError(f"{domain.name} does not support original analysis")
    root = output_dir or domain.config_root.parent / "results" / "original"
    importlib.import_module(domain.original_render_module).render(
        experiments, root=root
    )


def summarize_original(
    domain: DomainDefinition,
    experiments: list[str],
    *,
    output_dir: Path | None,
) -> None:
    if domain.original_summary_module is None:
        raise ValueError(f"{domain.name} has no original summaries")
    root = output_dir or domain.config_root.parent / "results" / "original"
    importlib.import_module(domain.original_summary_module).summarize(
        experiments, root=root
    )


def _validate_device(device: str | None) -> None:
    if device is not None and device not in {"cpu", "cuda:0"}:
        raise ValueError(f"unsupported device: {device}")
