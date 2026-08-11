"""Typer callbacks for the promoted DS1 original-source domain."""

from pathlib import Path
from typing import Annotated

import typer

from exp.cli_support import AtomicRuns, ExcludedAtomicRuns, Experiments, Overrides, cli_errors
from exp.domain import DomainDefinition, RunOrder
from exp.original import promoted_cli


DEFINITION = DomainDefinition(name="ds1_original", config_root=Path("exp/ds1_original/config"), spec_module="exp.ds1_original.spec", executor_module="exp.ds1_original.executor", analysis_module="exp.ds1_original.analyze.render")


@cli_errors
def plan(experiment: Experiments = None, all_experiments: Annotated[bool, typer.Option("--all")] = False, atomic_run: AtomicRuns = None, exclude_atomic_run: ExcludedAtomicRuns = None, seed_set: Annotated[str | None, typer.Option("--seed-set")] = None, seed: Annotated[str | None, typer.Option("--seed")] = None, device: Annotated[str | None, typer.Option("--device")] = None, override_values: Overrides = None, order: Annotated[RunOrder, typer.Option("--order")] = RunOrder.CATALOG_FIRST):
    promoted_cli.plan(DEFINITION, experiment=experiment, all_experiments=all_experiments, atomic_run=atomic_run, exclude_atomic_run=exclude_atomic_run, seed_set=seed_set, seed=seed, device=device, override_values=override_values, order=order)


@cli_errors
def run(experiment: Experiments = None, all_experiments: Annotated[bool, typer.Option("--all")] = False, atomic_run: AtomicRuns = None, exclude_atomic_run: ExcludedAtomicRuns = None, seed_set: Annotated[str | None, typer.Option("--seed-set")] = None, seed: Annotated[str | None, typer.Option("--seed")] = None, device: Annotated[str | None, typer.Option("--device")] = None, override_values: Overrides = None, order: Annotated[RunOrder, typer.Option("--order")] = RunOrder.CATALOG_FIRST, dry_run: Annotated[bool, typer.Option("--dry-run")] = False, progress: Annotated[str, typer.Option("--progress")] = "auto", progress_every: Annotated[int, typer.Option("--progress-every")] = 10, tracking_uri: Annotated[str | None, typer.Option("--tracking-uri")] = None):
    promoted_cli.run(DEFINITION, experiment=experiment, all_experiments=all_experiments, atomic_run=atomic_run, exclude_atomic_run=exclude_atomic_run, seed_set=seed_set, seed=seed, device=device, override_values=override_values, order=order, dry_run=dry_run, progress=progress, progress_every=progress_every, tracking_uri=tracking_uri)


@cli_errors
def analyze(experiment: Experiments = None, all_experiments: Annotated[bool, typer.Option("--all")] = False, tracking_uri: Annotated[str | None, typer.Option("--tracking-uri")] = None, error_style: Annotated[str, typer.Option("--error-style")] = "band", output_dir: Annotated[Path | None, typer.Option("--output-dir")] = None, seed: Annotated[int | None, typer.Option("--seed")] = None, summary: Annotated[bool, typer.Option("-s", "--summary")] = False):
    promoted_cli.analyze(DEFINITION, experiment=experiment, all_experiments=all_experiments, tracking_uri=tracking_uri, error_style=error_style, output_dir=output_dir, seed=seed, summary=summary)
