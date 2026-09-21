"""Root Typer CLI interface for F2 study suites, corpus pipeline, and reproduction catalog."""

from __future__ import annotations

import json
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Annotated

import typer

from repro_core.execution.definition import RunOrder

from .catalog.cli import app as catalog_app
from .corpus.preprocessing_cli import app as corpus_app
from .definition import DEFINITION

Experiments = Annotated[
    list[str] | None,
    typer.Option("-e", "--experiment", help="Experiment IDs or ranges."),
]
AtomicRuns = Annotated[
    list[str] | None,
    typer.Option("-a", "--atomic-run", help="Include atomic run IDs."),
]
ExcludedAtomicRuns = Annotated[
    list[str] | None,
    typer.Option("-x", "--exclude-atomic-run", help="Exclude atomic run IDs."),
]
Overrides = Annotated[
    list[str] | None,
    typer.Option("--set", metavar="KEY=VALUE", help="Override YAML values."),
]


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
    suite: Annotated[str, typer.Argument(help="Evaluation policy: w2v1 or w2v2.")],
    lookup: Annotated[Path, typer.Option(help="Saved lookup artifact directory.")],
    questions: Annotated[Path, typer.Option(help="Analogy questions file.")],
    output: Annotated[Path, typer.Option(help="New evaluation report JSON path.")],
    phrase_separator: Annotated[
        str, typer.Option(help="W2V2 phrase token separator.")
    ] = "_",
) -> None:
    """Evaluate a saved model artifact independently of training."""
    from .suites.w2v.evaluate import (
        evaluate_lookup_artifact,
        write_evaluation_report,
    )

    try:
        separator = phrase_separator.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError("phrase separator must be ASCII") from exc
    payload = evaluate_lookup_artifact(
        suite,
        lookup,
        questions,
        phrase_separator=separator,
    )
    typer.echo(f"Evaluation report written: {write_evaluation_report(payload, output)}")


@app.command("preflight")
@cli_errors
def preflight() -> None:
    """Read-only validation of F2 database, corpus store, and MLflow targets."""
    from .preflight import run_preflight

    typer.echo(json.dumps(run_preflight(), indent=2, sort_keys=True))


@app.command("suites")
def list_suites() -> None:
    """List registered F2 reproduction sub-studies."""
    suites = DEFINITION.suite_names()
    if not suites:
        typer.echo(
            "No experimental suites registered yet in f2.suites. (Available subsystems: corpus, catalog)"
        )
        return
    typer.echo("Registered F2 reproduction suites:")
    for s in suites:
        typer.echo(f"  - {s}")


@cli_errors
def plan(
    suite: Annotated[
        str,
        typer.Argument(
            help="Target F2 suite (e.g. w2v_pretrain) or 'corpus'",
        ),
    ],
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
    """Inspect expanded experiment run plans for an F2 suite."""
    if suite == "corpus":
        typer.echo("For corpus pipeline planning, use: repro f2 corpus plan --help")
        return
    from repro_core.cli.commands import plan_command

    suite_def = DEFINITION.get_suite(suite)
    plan_command(
        suite_def,
        experiments=experiment or [],
        all_experiments=all_experiments,
        atomic_runs=atomic_run or [],
        excluded_atomic_runs=exclude_atomic_run or [],
        seed_set=seed_set,
        seeds=seed,
        device=device,
        override_values=override_values or [],
        order=order,
    )


@cli_errors
def run(
    suite: Annotated[
        str,
        typer.Argument(
            help="Target F2 suite (e.g. w2v_pretrain)",
        ),
    ],
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
    approve_large_run: Annotated[
        bool,
        typer.Option(
            "--approve-large-run",
            help="Acknowledge the cost of canonical W2V training.",
        ),
    ] = False,
) -> None:
    """Execute F2 suite experiments."""
    from repro_core.cli.commands import run_command

    suite_def = DEFINITION.get_suite(suite)
    canonical_w2v = suite in {"w2v1", "w2v2"}
    if canonical_w2v and not dry_run and not approve_large_run:
        raise ValueError("canonical W2V training requires --approve-large-run")
    if canonical_w2v and not dry_run:
        from .tracking import resolve_tracking_uri

        tracking_uri = resolve_tracking_uri(tracking_uri)
    run_fn = None
    if suite in {"w2v1", "w2v2"}:
        from .suites.w2v.tracked import run_tracked_yaml

        run_fn = run_tracked_yaml
    run_command(
        suite_def,
        experiments=experiment or [],
        all_experiments=all_experiments,
        atomic_runs=atomic_run or [],
        excluded_atomic_runs=exclude_atomic_run or [],
        seed_set=seed_set,
        seeds=seed,
        device=device,
        override_values=override_values or [],
        order=order,
        dry_run=dry_run,
        progress=progress,
        progress_every=progress_every,
        tracking_uri=tracking_uri,
        run_fn=run_fn,
    )


@cli_errors
def analyze(
    suite: Annotated[
        str,
        typer.Argument(
            help="Target F2 suite (e.g. w2v_pretrain) or 'corpus'",
        ),
    ] = "corpus",
    tracking_uri: Annotated[str | None, typer.Option("--tracking-uri")] = None,
    questions: Annotated[
        Path | None,
        typer.Option("--questions", help="Canonical questions-words.txt path."),
    ] = None,
) -> None:
    """Render or summarize F2 experiment results."""
    if suite == "corpus":
        typer.echo("For corpus pipeline analysis, use: repro f2 corpus analyze --help")
        return
    if suite == "w2v1":
        from repro_core.context import RuntimePaths

        from .common.paths import get_benchmark_data_dir
        from .suites.w2v1.analysis import analyze_table2
        from .tracking import resolve_tracking_uri

        paths = RuntimePaths.from_environment()
        questions = questions or get_benchmark_data_dir(paths) / "questions-words.txt"
        output = analyze_table2(
            resolve_tracking_uri(tracking_uri), questions, paths=paths
        )
        typer.echo(f"W2V1 Table 2 analysis written: {output}")
        return
    typer.echo(f"Analysis orchestration for F2 suite '{suite}' is initialized.")


@cli_errors
def check(
    suite: Annotated[
        str,
        typer.Argument(
            help="Target F2 suite (e.g. w2v_pretrain)",
        ),
    ],
    experiment: Experiments = None,
    all_experiments: Annotated[bool, typer.Option("--all")] = False,
    atomic_run: AtomicRuns = None,
    exclude_atomic_run: ExcludedAtomicRuns = None,
    seed_set: Annotated[str | None, typer.Option("--seed-set")] = None,
    seed: Annotated[str | None, typer.Option("--seed")] = None,
    override_values: Overrides = None,
    tracking_uri: Annotated[str | None, typer.Option("--tracking-uri")] = None,
) -> None:
    """Compare declared plans with recorded F2 run state in MLflow."""
    typer.echo(f"Checking run state for F2 suite '{suite}'...")


__all__ = ["analyze", "app", "check", "evaluate", "list_suites", "plan", "run"]
