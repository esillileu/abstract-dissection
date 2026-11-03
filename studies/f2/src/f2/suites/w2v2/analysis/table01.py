"""Table 1 word-analogy evaluation over durable W2V2 MLflow artifacts."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from repro_core.context import RuntimePaths

from f2.storage.analysis_cache import LocalResults, unavailable_report
from f2.suites.common.w2v.analysis import (
    W2VTableEvaluationCache,
    format_accuracy_and_time,
    format_mean_sd,
    observed_training_seconds,
    write_run_records_csv,
    write_table_summary,
)
from f2.suites.common.w2v.artifacts import load_lookup_artifact
from f2.suites.common.w2v.evaluation import evaluate_analogies, parse_analogy_questions
from .. import paper

_SEEDS = (1, 7, 19)
_OBJECTIVES = (
    "neg5-no-subsampling",
    "neg15-no-subsampling",
    "hs-no-subsampling",
    "nce5-no-subsampling",
    "neg5-subsampling",
    "neg15-subsampling",
    "hs-subsampling",
)


@dataclass(frozen=True)
class Table1RunResult:
    condition: str
    seed: int
    mlflow_run_id: str
    total_accuracy_percent: float
    semantic_accuracy_percent: float
    syntactic_accuracy_percent: float
    observed_training_seconds: float
    included_questions: int
    total_questions: int


def _is_table1_run(record: dict[str, Any]) -> bool:
    tags = record.get("tags") or {}
    if tags.get("study") == "table1" or tags.get("experiment_id") == "e01":
        return True
    art_root = record.get("artifact_root") or ""
    if "table1" in art_root or "e01" in art_root:
        return True
    cond = tags.get("condition") or tags.get("variant") or ""
    return any(obj in cond for obj in ("neg5", "neg15", "hs", "nce5"))


def _extract_objective_condition(record: dict[str, Any]) -> str | None:
    tags = record.get("tags") or {}
    cond = tags.get("condition") or tags.get("variant") or ""
    for obj in _OBJECTIVES:
        if obj in cond:
            return obj
    return None


def analyze_table1(
    runs: list[Any],
    questions_path: Path | None,
    *,
    output_dir: Path,
    corpus_source: str = "news-1b",
    paths: RuntimePaths | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = paths or RuntimePaths.from_environment()

    if questions_path is None or not questions_path.is_file():
        return unavailable_report(
            output_dir,
            paper_reference=paper.SOURCE,
            table="1",
            reasons=["Analogies benchmark questions file questions-words.txt is missing."],
        )

    questions_bytes = questions_path.read_bytes()
    questions = parse_analogy_questions(questions_bytes.splitlines())
    questions_sha256 = hashlib.sha256(questions_bytes).hexdigest()

    cache_dir = paths.analysis_cache("f2", "w2v2") / "eval_cache"
    eval_cache = W2VTableEvaluationCache(cache_dir)

    matching_runs = [r for r in runs if _is_table1_run(r)]
    if not matching_runs:
        return unavailable_report(
            output_dir,
            paper_reference=paper.SOURCE,
            table="1",
            reasons=[f"No matching completed W2V2 Table 1 runs found for corpus {corpus_source}."],
        )

    run_results: list[Table1RunResult] = []
    for r in matching_runs:
        run_id = r.get("run_id")
        cond = _extract_objective_condition(r)
        if not cond or not run_id:
            continue
        seed = int((r.get("tags") or {}).get("seed", 1))
        art_root = Path(r.get("artifact_root", ""))
        lookup_dir = art_root / "lookup"
        if not lookup_dir.is_dir():
            lookup_dir = art_root

        cached = eval_cache.load(
            run_id,
            table="table01",
            condition=cond,
            questions_sha256=questions_sha256,
        )
        if cached is not None:
            res_dict = cached
        else:
            try:
                lookup = load_lookup_artifact(lookup_dir)
                eval_res = evaluate_analogies(lookup, questions)
                res_dict = {
                    "total_accuracy_percent": eval_res.total_accuracy_percent,
                    "semantic_accuracy_percent": eval_res.semantic_accuracy_percent,
                    "syntactic_accuracy_percent": eval_res.syntactic_accuracy_percent,
                    "included_questions": eval_res.included_questions,
                    "total_questions": eval_res.total_questions,
                }
                eval_cache.store(
                    run_id,
                    table="table01",
                    condition=cond,
                    questions_sha256=questions_sha256,
                    result=res_dict,
                )
            except Exception:
                continue

        run_results.append(
            Table1RunResult(
                condition=cond,
                seed=seed,
                mlflow_run_id=run_id,
                total_accuracy_percent=res_dict["total_accuracy_percent"],
                semantic_accuracy_percent=res_dict["semantic_accuracy_percent"],
                syntactic_accuracy_percent=res_dict["syntactic_accuracy_percent"],
                observed_training_seconds=observed_training_seconds(art_root),
                included_questions=res_dict["included_questions"],
                total_questions=res_dict["total_questions"],
            )
        )

    if not run_results:
        return unavailable_report(
            output_dir,
            paper_reference=paper.SOURCE,
            table="1",
            reasons=["No runs could be evaluated successfully."],
        )

    # Group by condition
    by_condition: dict[str, list[Table1RunResult]] = {}
    for res in run_results:
        by_condition.setdefault(res.condition, []).append(res)

    headers = [
        "Objective / Conditioning",
        "Observed Time",
        "Syntactic Acc % (Mean ± SD)",
        "Semantic Acc % (Mean ± SD)",
        "Total Acc % (Mean ± SD)",
        "Paper Reference (Time, Syn, Sem, Total)",
    ]
    rows = []
    for cond in _OBJECTIVES:
        items = by_condition.get(cond, [])
        ref = paper.TABLE1.get(cond, ("-", "-", "-", "-"))
        ref_str = f"{ref[0]}m, {ref[1]}%, {ref[2]}%, {ref[3]}%"
        if not items:
            rows.append([cond, "N/A", "N/A", "N/A", "N/A", ref_str])
            continue
        time_str = format_accuracy_and_time(
            [it.total_accuracy_percent for it in items],
            [it.observed_training_seconds for it in items],
        ).split("(")[-1].rstrip(")") if items else "N/A"
        syn_str = format_mean_sd([it.syntactic_accuracy_percent for it in items])
        sem_str = format_mean_sd([it.semantic_accuracy_percent for it in items])
        tot_str = format_mean_sd([it.total_accuracy_percent for it in items])
        rows.append([cond, time_str, syn_str, sem_str, tot_str, ref_str])

    write_table_summary(
        output_dir,
        title="W2V2 Table 1: Word-Analogy Accuracy and Training Time by Objective",
        paper_reference=paper.SOURCE,
        headers=headers,
        rows=rows,
    )
    records_rows = [
        {
            "condition": r.condition,
            "seed": r.seed,
            "mlflow_run_id": r.mlflow_run_id,
            "total_accuracy_percent": r.total_accuracy_percent,
            "semantic_accuracy_percent": r.semantic_accuracy_percent,
            "syntactic_accuracy_percent": r.syntactic_accuracy_percent,
            "observed_training_seconds": r.observed_training_seconds,
            "included_questions": r.included_questions,
            "total_questions": r.total_questions,
        }
        for r in run_results
    ]
    write_run_records_csv(output_dir / "runs.csv", records_rows)
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "paper": paper.SOURCE,
                "table": "1",
                "corpus": corpus_source,
                "results": records_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return output_dir


def analyze_table1_sources(
    cache_root: Path,
    questions_path: Path | None,
    *,
    corpus_source: str | None = None,
    paths: RuntimePaths | None = None,
) -> list[Path]:
    paths = paths or RuntimePaths.from_environment()
    runs_file = cache_root / "runs.json"
    if not runs_file.is_file():
        return []
    runs = json.loads(runs_file.read_text(encoding="utf-8"))
    c_source = corpus_source or "news-1b"
    out = paths.analysis_results("f2", "w2v2", "table01", c_source)
    return [analyze_table1(runs, questions_path, output_dir=out, corpus_source=c_source, paths=paths)]


__all__ = ["Table1RunResult", "analyze_table1", "analyze_table1_sources"]
