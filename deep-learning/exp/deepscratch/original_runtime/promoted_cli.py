"""Command implementations shared by promoted original domains."""

from __future__ import annotations

from pathlib import Path

from exp.commands import analyze_command, plan_command, run_command
from exp.domain import DomainDefinition, RunOrder


def plan(domain: DomainDefinition, *, experiment, all_experiments, atomic_run, exclude_atomic_run, seed_set, seed, device, override_values, order: RunOrder):
    plan_command(domain, experiments=experiment or [], all_experiments=all_experiments, atomic_runs=atomic_run or [], excluded_atomic_runs=exclude_atomic_run or [], seed_set=seed_set, seeds=seed, device=device, override_values=override_values or [], order=order)


def run(domain: DomainDefinition, *, experiment, all_experiments, atomic_run, exclude_atomic_run, seed_set, seed, device, override_values, order: RunOrder, dry_run, progress, progress_every, tracking_uri):
    run_command(domain, experiments=experiment or [], all_experiments=all_experiments, atomic_runs=atomic_run or [], excluded_atomic_runs=exclude_atomic_run or [], seed_set=seed_set, seeds=seed, device=device, override_values=override_values or [], order=order, dry_run=dry_run, progress=progress, progress_every=progress_every, tracking_uri=tracking_uri, original=False, force=False, output_dir=None)


def analyze(domain: DomainDefinition, *, experiment, all_experiments, tracking_uri, error_style, output_dir: Path | None, seed, summary):
    analyze_command(domain, experiments=experiment or [], all_experiments=all_experiments, tracking_uri=tracking_uri, error_style=error_style, output_dir=output_dir, seed=seed, summary=summary, original=False)
