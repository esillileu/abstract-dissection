"""Table 1 word-analogy evaluation over durable W2V2 MLflow artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from f2.storage.analysis_cache import unavailable_report
from f2.suites.common.w2v.analysis import (
    W2VTableEvaluationCache,
    format_accuracy_and_time,
    format_mean_sd,
    observed_training_seconds,
    write_run_records_csv,
    write_table_summary,
)
from f2.suites.common.w2v.evaluation import evaluate_analogies, parse_analogy_questions
from f2.suites.common.w2v.lookup import load_lookup_artifact
from repro_core.context import RuntimePaths

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


def _is_table1_run(record: dict[str, Any], corpus_source: str) -> bool:
    tags = record.get("tags") or {}
    spec = tags.get("experiment_spec.id")
    study = tags.get("study") or tags.get("f2.experiment_id")
    if spec != "w2v2-word-skipgram-1b-objectives" and study not in ("table1", "e01"):
        return False
    variant = tags.get("implementation.variant") or tags.get("variant") or ""
    corpus_id = tags.get("corpus.id") or ""
    if corpus_id:
        return corpus_id == corpus_source
    if "--" in variant:
        return variant.split("--")[0] == corpus_source
    return False


def _extract_objective_condition(record: dict[str, Any]) -> str | None:
    tags = record.get("tags") or {}
    cond = (
        tags.get("condition")
        or tags.get("variant")
        or tags.get("implementation.variant")
        or ""
    )
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
            "Analogies benchmark questions file questions-words.txt is missing.",
        )

    questions_bytes = questions_path.read_bytes()
    questions = parse_analogy_questions(questions_bytes.splitlines())
    questions_sha256 = hashlib.sha256(questions_bytes).hexdigest()

    cache_dir = paths.analysis_cache_root / "f2/w2v2/analysis/table1"
    eval_cache = W2VTableEvaluationCache(
        cache_dir,
        protocol="w2v2-table1",
        result_cls=dict,
        questions_sha256=questions_sha256,
    )

    matching_runs = [r for r in runs if _is_table1_run(r, corpus_source)]
    if not matching_runs:
        return unavailable_report(
            output_dir,
            f"No matching completed W2V2 Table 1 runs found for corpus {corpus_source}.",
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

        cached = eval_cache.load(run_id)
        if cached is not None:
            res_dict = cached
        else:
            try:
                lookup = load_lookup_artifact(lookup_dir)
                eval_res = evaluate_analogies(lookup, questions)
                res_dict = {
                    "total_accuracy_percent": 100.0 * (eval_res.overall.score or 0.0),
                    "semantic_accuracy_percent": 100.0
                    * (eval_res.semantic.score or 0.0),
                    "syntactic_accuracy_percent": 100.0
                    * (eval_res.syntactic.score or 0.0),
                    "included_questions": eval_res.overall.valid_count,
                    "total_questions": eval_res.overall.total_count,
                }
                eval_cache.store(run_id, res_dict)
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
            "No runs could be evaluated successfully.",
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
        _, time_str = format_accuracy_and_time(items) if items else ("—", "N/A")
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
    sources = [corpus_source] if corpus_source else ["wmt", "lm1b", "umbc"]
    results = []
    for c_source in sources:
        out = paths.analysis_results_root / f"f2/w2v2/table01/{c_source}"
        results.append(
            analyze_table1(
                runs,
                questions_path,
                output_dir=out,
                corpus_source=c_source,
                paths=paths,
            )
        )
    return results


__all__ = ["Table1RunResult", "analyze_table1", "analyze_table1_sources"]
