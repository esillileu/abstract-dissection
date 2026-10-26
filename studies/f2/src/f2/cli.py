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
    override: Annotated[
        bool,
        typer.Option(
            "--override",
            help="Re-run and publish a new MLflow run, preserving existing runs.",
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

        if override:

            def run_fn(*args, **kwargs):
                return run_tracked_yaml(*args, override=True, **kwargs)
        else:
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
            help="Target F2 suite (e.g. w2v1, w2v2) or 'corpus'",
        ),
    ] = "corpus",
    tracking_uri: Annotated[str | None, typer.Option("--tracking-uri")] = None,
    questions: Annotated[
        Path | None,
        typer.Option("--questions", help="Canonical questions-words.txt path."),
    ] = None,
    phrase_questions: Annotated[
        Path | None,
        typer.Option(
            "--phrase-questions", help="Canonical questions-phrases.txt path."
        ),
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
) -> None:
    """Render or summarize F2 experiment results."""
    if suite == "corpus":
        typer.echo("For corpus pipeline analysis, use: repro f2 corpus analyze --help")
        return
    if suite == "w2v2":
        from repro_core.context import RuntimePaths

        from .suites.w2v.cache import download_results, read_records
        from .suites.w2v2.analysis import generate_report
        from .tracking import resolve_tracking_uri

        paths = RuntimePaths.from_environment()
        records = read_records(
            download_results(resolve_tracking_uri(tracking_uri), "w2v2", paths)
        )
        cache_dir = paths.analysis_cache("f2", "w2v2")
        benchmarks = paths.inputs("f2", "benchmarks")
        output = generate_report(
            records=records,
            paths=paths,
            word_questions=questions
            or (
                cache_dir / "evaluation/questions-words.txt"
                if (cache_dir / "evaluation/questions-words.txt").is_file()
                else benchmarks / "questions-words.txt"
            ),
            phrase_questions=phrase_questions
            or (
                cache_dir / "evaluation/questions-phrases.txt"
                if (cache_dir / "evaluation/questions-phrases.txt").is_file()
                else benchmarks / "questions-phrases.txt"
            ),
            result_source=f"MLflow f2.w2v2 ({resolve_tracking_uri(tracking_uri)})",
        )
        typer.echo(f"W2V2 report written: {output / 'summary.md'}")
        return
    if suite == "w2v1":
        from repro_core.context import RuntimePaths

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
        from .suites.w2v.cache import download_results

        cache_root = download_results(resolve_tracking_uri(tracking_uri), "w2v1", paths)
        if table == 7:
            questions = (
                questions
                or cache_root
                / "evaluation/msr_sentence_completion"
                / "Holmes.lm_format.questions.txt"
            )
            analyzer = analyze_table7_sources
            outputs = analyzer(
                cache_root,
                questions,
                paths=paths,
            )
        else:
            benchmarks = paths.inputs("f2", "benchmarks")
            questions = (
                questions
                or (
                    benchmarks / "questions-words.txt"
                    if (benchmarks / "questions-words.txt").is_file()
                    else None
                )
                or next(
                    (
                        p
                        for p in cache_root.glob("*/evaluation/questions-words.txt")
                        if p.is_file()
                    ),
                    cache_root / "evaluation/questions-words.txt",
                )
            )
            if table == 2:
                analyzer = analyze_table2_sources
            elif table == 3:
                outputs = analyze_table3_sources(
                    cache_root,
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
                    cache_root,
                    questions,
                    corpus_source=corpus,
                    paths=paths,
                )
                for output in outputs:
                    typer.echo(f"W2V1 Table {table} analysis written: {output}")
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
        if r.info.status == "FINISHED"
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


@app.command("phrase-preflight")
@cli_errors
def phrase_preflight(
    config: Path,
    atomic_run: Annotated[str, typer.Option("--atomic-run")],
    source: Annotated[Path | None, typer.Option("--source")] = None,
    prepare: Annotated[
        bool,
        typer.Option(
            "--prepare",
            help="Explicitly prepare phrases and measure vocabulary; never train.",
        ),
    ] = False,
) -> None:
    """Inspect Section 4.1 local inputs and resources without acquisition/training."""
    from repro_core.context.paths import RuntimePaths

    from .suites.w2v2.preflight import preflight, resolve_source
    from .suites.w2v2.spec import parse_run_spec

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
    offline: Annotated[
        bool,
        typer.Option(
            help="Analyze the existing local cache without contacting MLflow."
        ),
    ] = False,
    tracking_uri: Annotated[
        str | None, typer.Option(help="Read existing f2.w2v2 MLflow results.")
    ] = None,
    word_questions: Annotated[
        Path | None,
        typer.Option(help="Existing questions-words.txt; never downloaded."),
    ] = None,
    phrase_questions: Annotated[
        Path | None,
        typer.Option(help="Existing questions-phrases.txt; never downloaded."),
    ] = None,
) -> None:
    """Compare all six paper tables and Section 4.1 without training or publishing."""
    from repro_core.context import RuntimePaths

    from .suites.w2v.cache import download_results, read_records
    from .suites.w2v2.analysis import generate_report

    paths = RuntimePaths.from_environment()
    from .tracking import resolve_tracking_uri

    cache_dir = paths.analysis_cache("f2", "w2v2")
    records = (
        read_records(cache_dir)
        if offline
        else read_records(
            download_results(resolve_tracking_uri(tracking_uri), "w2v2", paths)
        )
    )
    benchmarks = paths.inputs("f2", "benchmarks")
    output = generate_report(
        records=records,
        paths=paths,
        word_questions=word_questions
        or (
            cache_dir / "evaluation/questions-words.txt"
            if (cache_dir / "evaluation/questions-words.txt").is_file()
            else benchmarks / "questions-words.txt"
        ),
        phrase_questions=phrase_questions
        or (
            cache_dir / "evaluation/questions-phrases.txt"
            if (cache_dir / "evaluation/questions-phrases.txt").is_file()
            else benchmarks / "questions-phrases.txt"
        ),
        result_source="offline; MLflow not queried" if offline else "MLflow f2.w2v2",
    )
    typer.echo(f"W2V2 report: {output / 'summary.md'}")
