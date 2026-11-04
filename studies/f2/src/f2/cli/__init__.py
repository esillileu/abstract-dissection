"""Root Typer CLI interface for F2 study suites, corpus pipeline, and reproduction catalog."""

from __future__ import annotations

import json
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Annotated

import typer

from repro_core.execution.definition import RunOrder

from ..catalog.cli import app as catalog_app
from ..corpus.cli import app as corpus_app
from ..execution.definitions import DEFINITION
from .analyze import analyze_command_impl, run_w2v2_analysis
from .check import check_command_impl
from .plan import (
    AtomicRuns,
    ExcludedAtomicRuns,
    Experiments,
    Overrides,
    plan_command_impl,
)
from .run import run_command_impl


def cli_errors[**P, T](function: Callable[P, T]) -> Callable[P, T]:
    """Wrap CLI callbacks to format value/runtime errors cleanly."""

    @wraps(function)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> T:
        try:
            return function(*args, **kwargs)
        except (ValueError, RuntimeError) as exc:
            raise typer.BadParameter(str(exc)) from None

    return wrapped


app = typer.Typer(
    name="f2",
    help="Word2Vec (2013) Paper Reproduction, Corpus Pipeline & Catalog.",
    no_args_is_help=True,
)

app.add_typer(corpus_app, name="corpus")
app.add_typer(catalog_app, name="catalog")


@app.command("evaluate")
@cli_errors
def evaluate(
    suite: Annotated[
        str,
        typer.Argument(
            help="Evaluation policy: w2v1-table2, w2v1-table4, w2v1-table7, w2v2-table3, or w2v2."
        ),
    ],
    lookup: Annotated[Path, typer.Option(help="Saved lookup artifact directory.")],
    questions: Annotated[
        Path,
        typer.Option(help="Evaluation benchmark file or directory."),
    ],
    output: Annotated[Path, typer.Option(help="New evaluation report JSON path.")],
    phrase_separator: Annotated[
        str, typer.Option(help="W2V2 phrase token separator.")
    ] = "_",
) -> None:
    """Evaluate a saved model artifact independently of training."""
    from ..suites.common.w2v.evaluate import (
        evaluate_lookup_artifact,
        write_evaluation_report,
    )

    try:
        separator = phrase_separator.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError("phrase separator must be ASCII") from exc
    payload = evaluate_lookup_artifact(
        suite, lookup, questions, phrase_separator=separator
    )
    typer.echo(f"Evaluation report written: {write_evaluation_report(payload, output)}")


@app.command("preflight")
@cli_errors
def preflight() -> None:
    """Read-only validation of F2 database, corpus store, and MLflow targets."""
    from ..preflight import run_preflight

    typer.echo(json.dumps(run_preflight(), indent=2, sort_keys=True))


@app.command("suites")
def list_suites() -> None:
    """List registered F2 reproduction sub-studies."""
    suites = DEFINITION.suite_names()
    if not suites:
        typer.echo("No experimental suites registered yet in f2.suites.")
        return
    typer.echo("Registered F2 reproduction suites:")
    for s in suites:
        typer.echo(f"  - {s}")


@cli_errors
def plan(
    suite: Annotated[str, typer.Argument(help="Target F2 suite or 'corpus'")],
    experiment: Experiments = None,
    all_experiments: Annotated[bool, typer.Option("--all")] = False,
    atomic_run: AtomicRuns = None,
    exclude_atomic_run: ExcludedAtomicRuns = None,
    seed_set: Annotated[str | None, typer.Option("--seed-set")] = None,
    seed: Annotated[str | None, typer.Option("--seed")] = None,
    device: Annotated[str | None, typer.Option("--device")] = None,
    override_values: Overrides = None,
    order: Annotated[RunOrder, typer.Option("--order")] = RunOrder.CATALOG_FIRST,
) -> None:
    plan_command_impl(
        suite,
        experiment=experiment,
        all_experiments=all_experiments,
        atomic_run=atomic_run,
        exclude_atomic_run=exclude_atomic_run,
        seed_set=seed_set,
        seed=seed,
        device=device,
        override_values=override_values,
        order=order,
    )


@cli_errors
def run(
    suite: Annotated[str, typer.Argument(help="Target F2 suite (e.g. w2v1, w2v2)")],
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
    progress: Annotated[bool, typer.Option("--progress")] = True,
    progress_every: Annotated[int | None, typer.Option("--progress-every")] = None,
    tracking_uri: Annotated[str | None, typer.Option("--tracking-uri")] = None,
    approve_large_run: Annotated[bool, typer.Option("--approve-large-run")] = False,
    override: Annotated[bool, typer.Option("--override")] = False,
) -> None:
    run_command_impl(
        suite,
        experiment=experiment,
        all_experiments=all_experiments,
        atomic_run=atomic_run,
        exclude_atomic_run=exclude_atomic_run,
        seed_set=seed_set,
        seed=seed,
        device=device,
        override_values=override_values,
        order=order,
        dry_run=dry_run,
        progress=progress,
        progress_every=progress_every,
        tracking_uri=tracking_uri,
        approve_large_run=approve_large_run,
        override=override,
    )


@app.command("analyze")
@cli_errors
def analyze(
    suite: Annotated[
        str | None, typer.Argument(help="Target F2 suite or 'corpus'")
    ] = None,
    tracking_uri: Annotated[str | None, typer.Option("--tracking-uri")] = None,
    questions: Annotated[Path | None, typer.Option("--questions")] = None,
    phrase_questions: Annotated[Path | None, typer.Option("--phrase-questions")] = None,
    corpus: Annotated[str | None, typer.Option("--corpus")] = None,
    table: Annotated[str | None, typer.Option("--table")] = None,
    msr_questions: Annotated[Path | None, typer.Option("--msr-questions")] = None,
    override: Annotated[bool, typer.Option("--override")] = False,
    offline: Annotated[bool, typer.Option("--offline")] = False,
) -> None:
    analyze_command_impl(
        suite=suite,
        tracking_uri=tracking_uri,
        questions=questions,
        phrase_questions=phrase_questions,
        corpus=corpus,
        table=table,
        msr_questions=msr_questions,
        override=override,
        offline=offline,
    )


@cli_errors
def check(
    suite: Annotated[str, typer.Argument(help="Target F2 suite (e.g. w2v1, w2v2)")],
    experiment: Experiments = None,
    all_experiments: Annotated[bool, typer.Option("--all")] = False,
    atomic_run: AtomicRuns = None,
    exclude_atomic_run: ExcludedAtomicRuns = None,
    seed_set: Annotated[str | None, typer.Option("--seed-set")] = None,
    seed: Annotated[str | None, typer.Option("--seed")] = None,
    override_values: Overrides = None,
    tracking_uri: Annotated[str | None, typer.Option("--tracking-uri")] = None,
) -> None:
    check_command_impl(
        suite,
        experiment=experiment,
        all_experiments=all_experiments,
        atomic_run=atomic_run,
        exclude_atomic_run=exclude_atomic_run,
        seed_set=seed_set,
        seed=seed,
        override_values=override_values,
        tracking_uri=tracking_uri,
    )


@app.command("phrase-preflight")
@cli_errors
def phrase_preflight(
    config: Path,
    atomic_run: Annotated[str, typer.Option("--atomic-run")],
    source: Annotated[Path | None, typer.Option("--source")] = None,
    prepare: Annotated[bool, typer.Option("--prepare")] = False,
) -> None:
    """Inspect Section 4.1 local inputs and resources without acquisition/training."""
    from repro_core.context.paths import RuntimePaths

    from ..suites.w2v2.preflight import preflight, resolve_source
    from ..suites.w2v2.spec import parse_run_spec

    paths = RuntimePaths.from_environment()
    spec = parse_run_spec(config, atomic_run_id=atomic_run)
    resolved = spec.to_executor_config()
    candidate = resolve_source(resolved, paths, source)
    typer.echo(
        json.dumps(
            preflight(
                resolved,
                candidate,
                paths.staging_root / "f2/preparation",
                staging_root=paths.staging_root,
                prepare=prepare,
            ),
            indent=2,
            sort_keys=True,
        )
    )


@app.command("w2v2-report")
@cli_errors
def w2v2_report(
    offline: Annotated[bool, typer.Option(help="Analyze without MLflow.")] = False,
    tracking_uri: Annotated[str | None, typer.Option(help="MLflow URI.")] = None,
    word_questions: Annotated[Path | None, typer.Option()] = None,
    phrase_questions: Annotated[Path | None, typer.Option()] = None,
    table: Annotated[int | None, typer.Option("--table")] = None,
    corpus: Annotated[str | None, typer.Option("--corpus")] = None,
    override: Annotated[bool, typer.Option("--override")] = False,
) -> None:
    """Compare all six paper tables and Section 4.1 without training."""
    output = run_w2v2_analysis(
        tracking_uri=tracking_uri,
        word_questions=word_questions,
        phrase_questions=phrase_questions,
        override=override,
        offline=offline,
        table=table,
        corpus=corpus,
    )
    typer.echo(f"W2V2 report: {output / 'summary.md'}")


__all__ = [
    "analyze",
    "app",
    "check",
    "evaluate",
    "list_suites",
    "plan",
    "run",
    "run_w2v2_analysis",
]
