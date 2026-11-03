"""Analysis command implementation and report orchestrators for W2V1/W2V2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer
from repro_io.checksum import sha256_file

from repro_core.context import RuntimePaths

from ..storage.analysis_cache import download_results, read_records
from ..suites.w2v2.analysis import EVALUATION_PROTOCOL, generate_report
from ..tracking import resolve_tracking_uri


def analyze_command_impl(
    suite: str | None = None,
    tracking_uri: str | None = None,
    questions: Path | None = None,
    phrase_questions: Path | None = None,
    corpus: str | None = None,
    table: int | None = None,
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
        output = run_w2v2_analysis(
            tracking_uri=tracking_uri,
            word_questions=questions,
            phrase_questions=phrase_questions,
            override=override,
            offline=offline,
            table=table,
            corpus=corpus,
        )
        typer.echo(f"W2V2 report written: {output / 'summary.md'}")
        return
    if suite == "w2v1":
        _analyze_w2v1(
            tracking_uri=tracking_uri,
            questions=questions,
            corpus=corpus,
            table=table,
            msr_questions=msr_questions,
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
    """Run W2V2 analysis or return existing result without MLflow access or recomputation."""
    paths = paths or RuntimePaths.from_environment()
    output = (
        paths.analysis_output("f2", "w2v2")
        if hasattr(paths, "analysis_output")
        else paths.cache_root / "f2/w2v2/analysis"
    )
    target_table = str(table).lower() if table is not None else None
    if target_table in {"section4", "4.1"}:
        target_table = "section4.1"
    target_corpus = str(corpus).lower() if corpus is not None else None
    check_output = output
    if target_corpus is not None:
        check_output = check_output / target_corpus
    if target_table is not None:
        table_dir = (
            f"table{target_table}" if target_table != "section4.1" else "section4_1"
        )
        check_output = check_output / table_dir

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

    cache_dir = paths.analysis_cache("f2", "w2v2")
    if offline:
        records = read_records(cache_dir)
        result_source = "offline; MLflow not queried"
    else:
        resolved_uri = resolve_tracking_uri(tracking_uri)
        records = read_records(
            download_results(resolved_uri, "w2v2", paths, table=table)
        )
        result_source = f"MLflow f2.w2v2 ({resolved_uri})"

    current_run_ids = sorted({r.get("run_id") for r in records if r.get("run_id")})
    summary_file = check_output / "summary.json"
    summary_md = check_output / "summary.md"
    if not override and summary_file.is_file() and summary_md.is_file():
        try:
            cached_payload = json.loads(summary_file.read_text(encoding="utf-8"))
            sig = cached_payload.get("analysis_signature")
            if isinstance(sig, dict):
                if _verify_sig(
                    sig, current_run_ids, target_table, target_corpus, w_q, p_q
                ):
                    return check_output
                raise ValueError(
                    f"Existing analysis in {check_output} corresponds to different runs/inputs. "
                    f"Use --override to overwrite previous analysis deliverables."
                )
            raise ValueError(
                f"Existing analysis in {check_output} lacks required signature. "
                f"Use --override to overwrite previous analysis deliverables."
            )
        except json.JSONDecodeError:
            pass

    return generate_report(
        records=records,
        paths=paths,
        word_questions=w_q,
        phrase_questions=p_q,
        result_source=result_source,
        table_filter=table,
        corpus_filter=corpus,
        use_evaluation_cache=True,
    )


def _eval_digest(path: Path | None) -> str | None:
    if path is not None and Path(path).is_file():
        try:
            return sha256_file(Path(path))
        except Exception:
            return None
    return None


def _verify_sig(
    sig: dict, current_run_ids: list[str], target_table, target_corpus, w_q, p_q
) -> bool:
    if not isinstance(sig, dict):
        return False
    for req in (
        "run_ids",
        "table_filter",
        "corpus_filter",
        "eval_datasets",
        "evaluation_protocol",
    ):
        if req not in sig:
            return False
    if sig.get("evaluation_protocol") != EVALUATION_PROTOCOL:
        return False
    if sig.get("run_ids") != current_run_ids:
        return False
    if (
        sig.get("table_filter") != target_table
        or sig.get("corpus_filter") != target_corpus
    ):
        return False
    eval_ds = sig.get("eval_datasets", {})
    if not isinstance(eval_ds, dict):
        return False
    w_sha = _eval_digest(w_q)
    p_sha = _eval_digest(p_q)
    if w_sha and eval_ds.get("word", {}).get("sha256") != w_sha:
        return False
    if p_sha and eval_ds.get("phrase", {}).get("sha256") != p_sha:
        return False
    return True


def _analyze_w2v1(tracking_uri, questions, corpus, table, msr_questions, offline):
    from ..suites.w2v1.analysis import analyze_table2_sources
    from ..suites.w2v1.analysis.table03 import analyze_table3_sources
    from ..suites.w2v1.analysis.table04 import analyze_table4_sources
    from ..suites.w2v1.analysis.table05 import analyze_table5_sources
    from ..suites.w2v1.analysis.table06 import analyze_table6_sources
    from ..suites.w2v1.analysis.table07 import analyze_table7_sources

    paths = RuntimePaths.from_environment()
    tbl = 2 if table is None else table
    if tbl not in (2, 3, 4, 5, 6, 7):
        raise ValueError("W2V1 analysis table must be 2, 3, 4, 5, 6, or 7")
    if msr_questions is not None and tbl != 3:
        raise ValueError("--msr-questions applies only to Table 3")

    if offline:
        cache_root = paths.analysis_cache("f2", "w2v1")
        if not (cache_root / "runs.json").is_file():
            typer.echo("W2V1 offline analysis cache unavailable: missing runs.json")
            return
    else:
        cache_root = download_results(resolve_tracking_uri(tracking_uri), "w2v1", paths)

    if tbl == 7:
        q_file = (
            questions
            or cache_root
            / "evaluation/msr_sentence_completion/Holmes.lm_format.questions.txt"
        )
        outputs = analyze_table7_sources(cache_root, q_file, paths=paths)
    else:
        benchmarks = paths.inputs("f2", "benchmarks")
        q_file = questions or (
            benchmarks / "questions-words.txt"
            if (benchmarks / "questions-words.txt").is_file()
            else None
        )
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
