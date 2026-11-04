"""Analysis command implementation and report orchestrators for W2V1/W2V2."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import typer

from repro_core.context import RuntimePaths

from ..storage.analysis_cache import download_results
from ..tracking import resolve_tracking_uri


def analyze_command_impl(
    suite: str | None = None,
    tracking_uri: str | None = None,
    questions: Path | None = None,
    phrase_questions: Path | None = None,
    corpus: str | None = None,
    table: int | str | None = None,
    msr_questions: Path | None = None,
    override: bool = False,
    offline: bool = False,
) -> None:
    """Render or summarize F2 experiment results."""
    target_suite = (suite or "").lower()
    if target_suite == "corpus":
        typer.echo("For corpus pipeline analysis, use: repro f2 corpus analyze --help")
        return
    if target_suite in ("w2v2", ""):
        _analyze_w2v2(
            tracking_uri=tracking_uri,
            word_questions=questions,
            phrase_questions=phrase_questions,
            table=table,
            corpus=corpus,
            override=override,
            offline=offline,
        )
        return
    if suite == "w2v1":
        _analyze_w2v1(
            tracking_uri=tracking_uri,
            questions=questions,
            corpus=corpus,
            table=table,
            msr_questions=msr_questions,
            override=override,
            offline=offline,
        )
        return
    typer.echo(f"Analysis orchestration for F2 suite '{suite}' is initialized.")


def run_w2v2_analysis(
    *,
    paths: Any | None = None,
    tracking_uri: str | None = None,
    word_questions: Path | None = None,
    phrase_questions: Path | None = None,
    override: bool = False,
    offline: bool = False,
    table: int | str | None = None,
    corpus: str | None = None,
) -> Path:
    """Facade for W2V2 analysis matching stable paths."""
    outputs = _analyze_w2v2(
        tracking_uri=tracking_uri,
        word_questions=word_questions,
        phrase_questions=phrase_questions,
        table=table,
        corpus=corpus,
        override=override,
        offline=offline,
    )
    return (
        outputs[0]
        if outputs
        else (paths or RuntimePaths.from_environment()).analysis_results("f2", "w2v2")
    )


def _analyze_w2v2(
    *,
    tracking_uri: str | None = None,
    word_questions: Path | None = None,
    phrase_questions: Path | None = None,
    table: int | str | None = None,
    corpus: str | None = None,
    override: bool = False,
    offline: bool = False,
) -> list[Path]:
    from ..suites.w2v2.analysis import (
        analyze_section04_1_sources,
        analyze_table1_sources,
        analyze_table2_sources,
        analyze_table3_sources,
        analyze_table4_sources,
        analyze_table5_sources,
        analyze_table6_sources,
    )

    paths = RuntimePaths.from_environment()
    tbl_str = str(table).lower() if table is not None else None
    if tbl_str in {"section4", "4.1"}:
        tbl_str = "section04_1"
    elif tbl_str in {"1", "table1", "table01"}:
        tbl_str = "1"
    elif tbl_str in {"2", "table2", "table02"}:
        tbl_str = "2"
    elif tbl_str in {"3", "table3", "table03"}:
        tbl_str = "3"
    elif tbl_str in {"4", "table4", "table04"}:
        tbl_str = "4"
    elif tbl_str in {"5", "table5", "table05"}:
        tbl_str = "5"
    elif tbl_str in {"6", "table6", "table06"}:
        tbl_str = "6"

    if override:
        cache = paths.analysis_cache_root / "f2/w2v2/analysis"
        if tbl_str is None:
            shutil.rmtree(cache, ignore_errors=True)
        elif tbl_str in {"1", "3", "section04_1"}:
            name = f"table{tbl_str}" if tbl_str != "section04_1" else tbl_str
            shutil.rmtree(cache / name, ignore_errors=True)
    if offline:
        cache_root = paths.analysis_cache("f2", "w2v2")
        if not (cache_root / "runs.json").is_file():
            typer.echo("W2V2 offline analysis cache unavailable: missing runs.json")
            return []
    else:
        cache_root = download_results(resolve_tracking_uri(tracking_uri), "w2v2", paths)

    benchmarks = paths.inputs("f2", "benchmarks")
    w_q = word_questions or (
        benchmarks / "questions-words.txt"
        if (benchmarks / "questions-words.txt").is_file()
        else None
    )
    p_q = phrase_questions or (
        benchmarks / "questions-phrases.txt"
        if (benchmarks / "questions-phrases.txt").is_file()
        else None
    )

    analyzers = {
        "1": lambda: analyze_table1_sources(
            cache_root, w_q, corpus_source=corpus, paths=paths
        ),
        "2": lambda: analyze_table2_sources(
            cache_root, corpus_source=corpus, paths=paths
        ),
        "3": lambda: analyze_table3_sources(
            cache_root, p_q, corpus_source=corpus, paths=paths
        ),
        "4": lambda: analyze_table4_sources(
            cache_root, corpus_source=corpus, paths=paths
        ),
        "5": lambda: analyze_table5_sources(
            cache_root, corpus_source=corpus, paths=paths
        ),
        "6": lambda: analyze_table6_sources(
            cache_root, corpus_source=corpus, paths=paths
        ),
        "section04_1": lambda: analyze_section04_1_sources(
            cache_root, p_q, corpus_source=corpus, paths=paths
        ),
    }

    selected = [tbl_str] if tbl_str in analyzers else list(analyzers.keys())
    results: list[Path] = []
    for key in selected:
        outs = analyzers[key]()
        for out in outs:
            typer.echo(f"W2V2 {key} analysis written: {out}")
            results.append(out)
    return results


def _analyze_w2v1(
    tracking_uri,
    questions,
    corpus,
    table,
    msr_questions,
    override: bool = False,
    offline: bool = False,
):
    from ..suites.w2v1.analysis import analyze_table2_sources
    from ..suites.w2v1.analysis.table03 import analyze_table3_sources
    from ..suites.w2v1.analysis.table04 import analyze_table4_sources
    from ..suites.w2v1.analysis.table05 import analyze_table5_sources
    from ..suites.w2v1.analysis.table06 import analyze_table6_sources
    from ..suites.w2v1.analysis.table07 import analyze_table7_sources

    paths = RuntimePaths.from_environment()
    tbl = 2 if table is None else int(table)
    if tbl not in (2, 3, 4, 5, 6, 7):
        raise ValueError("W2V1 analysis table must be 2, 3, 4, 5, 6, or 7")
    if msr_questions is not None and tbl != 3:
        raise ValueError("--msr-questions applies only to Table 3")

    if override:
        cache = paths.analysis_cache_root / f"f2/w2v1/analysis/table{tbl}"
        if corpus and tbl in (3, 4, 5):
            cache /= corpus
        shutil.rmtree(cache, ignore_errors=True)
    if offline:
        cache_root = paths.analysis_cache("f2", "w2v1")
        if not (cache_root / "runs.json").is_file():
            typer.echo("W2V1 offline analysis cache unavailable: missing runs.json")
            return
    else:
        cache_root = download_results(resolve_tracking_uri(tracking_uri), "w2v1", paths)

    if tbl == 7:
        benchmarks = paths.inputs("f2", "benchmarks")
        default_q = (
            benchmarks / "msr_sentence_completion/Holmes.lm_format.questions.txt"
        )
        if not default_q.is_file():
            default_q = benchmarks / "Holmes.lm_format.questions.txt"
        if not default_q.is_file():
            default_q = (
                cache_root
                / "evaluation/msr_sentence_completion/Holmes.lm_format.questions.txt"
            )

        default_a = benchmarks / "msr_sentence_completion/Holmes.lm_format.answers.txt"
        if not default_a.is_file():
            default_a = benchmarks / "Holmes.lm_format.answers.txt"
        if not default_a.is_file():
            default_a = (
                cache_root
                / "evaluation/msr_sentence_completion/Holmes.lm_format.answers.txt"
            )

        q_file = questions or default_q
        outputs = analyze_table7_sources(
            cache_root,
            q_file,
            answers_path=default_a,
            paths=paths,
        )
    else:
        benchmarks = paths.inputs("f2", "benchmarks")
        default_q = benchmarks / "questions-words.txt"
        q_file = questions or (default_q if default_q.is_file() else None)
        analyzers = {
            2: analyze_table2_sources,
            4: analyze_table4_sources,
            5: analyze_table5_sources,
            6: analyze_table6_sources,
        }
        if tbl == 3:
            outputs = analyze_table3_sources(
                cache_root,
                q_file,
                msr_questions_path=msr_questions,
                corpus_source=corpus,
                paths=paths,
            )
            for out in outputs:
                typer.echo(f"W2V1 Table 3 analysis written: {out}")
            return
        outputs = analyzers[tbl](cache_root, q_file, corpus_source=corpus, paths=paths)
    for out in outputs:
        typer.echo(f"W2V1 Table {tbl} analysis written: {out}")


__all__ = ["analyze_command_impl", "run_w2v2_analysis"]
