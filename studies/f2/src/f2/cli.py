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
from .corpus.cli import app as corpus_app
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
    suite: Annotated[
        str,
        typer.Argument(
            help="Evaluation policy: w2v1-table2, w2v1-table4, w2v1-table7, or w2v2."
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
    corpus: Annotated[
        str | None,
        typer.Option(
            "--corpus",
            help="W2V1 corpus source: wmt, lm1b, umbc; fineweb for Tables 4/6. Omit to analyze all.",
        ),
    ] = None,
    table: Annotated[
        int, typer.Option("--table", help="W2V1 table number: 2, 3, 4, 5, 6, or 7.")
    ] = 2,
    msr_questions: Annotated[
        Path | None,
        typer.Option(
            "--msr-questions", help="Table 3 MSR syntactic word-analogy benchmark path."
        ),
    ] = None,
    publish: Annotated[
        bool,
        typer.Option(
            "--publish/--no-publish",
            help="Publish analysis results to canonical MLflow experiment f2.w2v1.analysis.",
        ),
    ] = False,
) -> None:
    """Render or summarize F2 experiment results."""
    if suite == "corpus":
        typer.echo("For corpus pipeline analysis, use: repro f2 corpus analyze --help")
        return
    if suite == "w2v1":
        from repro_core.context import RuntimePaths

        from .common.paths import get_benchmark_data_dir
        from .suites.w2v1.analysis import analyze_table2_sources
        from .suites.w2v1.table3 import analyze_table3_sources
        from .suites.w2v1.table4 import analyze_table4_sources
        from .suites.w2v1.table5 import analyze_table5_sources
        from .suites.w2v1.table6 import analyze_table6_sources
        from .suites.w2v1.table7 import analyze_table7_sources
        from .tracking import resolve_tracking_uri

        paths = RuntimePaths.from_environment()
        if table not in (2, 3, 4, 5, 6, 7):
            raise ValueError("W2V1 analysis table must be 2, 3, 4, 5, 6, or 7")
        if msr_questions is not None and table != 3:
            raise ValueError("--msr-questions applies only to Table 3")
        if table == 7:
            questions = (
                questions
                or get_benchmark_data_dir(paths)
                / "msr_sentence_completion"
                / "Holmes.lm_format.questions.txt"
            )
            analyzer = analyze_table7_sources
            outputs = analyzer(
                resolve_tracking_uri(tracking_uri),
                questions,
                paths=paths,
            )
        else:
            questions = (
                questions or get_benchmark_data_dir(paths) / "questions-words.txt"
            )
            if table == 2:
                analyzer = analyze_table2_sources
            elif table == 3:
                outputs = analyze_table3_sources(
                    resolve_tracking_uri(tracking_uri),
                    questions,
                    msr_questions_path=msr_questions,
                    corpus_source=corpus,
                    paths=paths,
                )
                for output in outputs:
                    typer.echo(f"W2V1 Table 3 analysis written: {output}")
                analyzer = None
            elif table == 4:
                analyzer = analyze_table4_sources
            elif table == 5:
                analyzer = analyze_table5_sources
            else:
                analyzer = analyze_table6_sources

            if analyzer is not None:
                outputs = analyzer(
                    resolve_tracking_uri(tracking_uri),
                    questions,
                    corpus_source=corpus,
                    paths=paths,
                )
                for output in outputs:
                    typer.echo(f"W2V1 Table {table} analysis written: {output}")

        if publish:
            from mlflow import MlflowClient

            from .suites.w2v1.publish_analysis import (
                ANALYSIS_EXPERIMENT_NAME,
                ensure_analysis_experiment,
                publish_table2_analysis,
                publish_table3_analysis,
                publish_table4_analysis,
                publish_table5_analysis,
                publish_table7_analysis,
            )

            client = MlflowClient(tracking_uri=resolve_tracking_uri(tracking_uri))
            exp_id = ensure_analysis_experiment(client)
            published = []
            if table == 7:
                published = publish_table7_analysis(
                    client, exp_id, paths.analysis_output("f2", "table7")
                )
            elif table == 5:
                published = publish_table5_analysis(
                    client, exp_id, paths.analysis_output("f2", "w2v1")
                )
            elif table == 4:
                published = publish_table4_analysis(
                    client, exp_id, paths.analysis_output("f2", "w2v1")
                )
            elif table == 3:
                published = publish_table3_analysis(
                    client, exp_id, paths.analysis_output("f2", "w2v1")
                )
            elif table == 2:
                published = publish_table2_analysis(
                    client, exp_id, paths.analysis_output("f2", "w2v1")
                )
            typer.echo(
                f"Published {len(published)} analysis runs to MLflow experiment '{ANALYSIS_EXPERIMENT_NAME}'"
            )
        return
    typer.echo(f"Analysis orchestration for F2 suite '{suite}' is initialized.")


@cli_errors
def check(
    suite: Annotated[
        str,
        typer.Argument(
            help="Target F2 suite (e.g. w2v1, w2v2)",
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
    from mlflow.tracking import MlflowClient

    from repro_core.execution import RunOptions, RunSelection
    from repro_core.execution.parsing import parse_overrides
    from repro_core.execution.planning import Planner

    from .definition import DEFINITION
    from .tracking import resolve_tracking_uri

    suite_def = DEFINITION.get_suite(suite)
    overrides = parse_overrides(override_values or [])
    plans = Planner(suite_def).build(
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
    uri = resolve_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=uri)
    experiment_name = f"f2.{suite}"
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        typer.echo(
            f"No MLflow experiment found for '{experiment_name}'. "
            f"All {len(plans)} planned runs are missing."
        )
        return

    runs = client.search_runs([exp.experiment_id], max_results=10000)
    completed_slots = {
        r.data.tags.get("planned_run_slot_id")
        or r.data.tags.get("f2.planned_run_slot_id")
        for r in runs
        if r.data.tags.get("result.durable_complete") == "true"
    }
    completed_count = 0
    missing_count = 0
    for plan in plans:
        spec = suite_def.load_run_spec(
            plan.path, atomic_run_id=plan.atomic_run_id, overrides=overrides
        )
        seeded_spec = spec.with_seed(plan.seed) if plan.seed is not None else spec
        slot_id = str(seeded_spec.identity.get("planned_run_slot_id", ""))
        if slot_id in completed_slots:
            completed_count += 1
        else:
            missing_count += 1
    typer.echo(
        f"f2/{suite}: planned={len(plans)} completed={completed_count} missing={missing_count}"
    )


__all__ = ["analyze", "app", "check", "evaluate", "list_suites", "plan", "run"]
