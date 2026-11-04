"""Table 3 phrase-analogy evaluation over durable W2V2 MLflow artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from f2.storage.analysis_cache import unavailable_report
from f2.suites.common.w2v.analysis import (
    W2VTableEvaluationCache,
    format_mean_sd,
    observed_training_seconds,
    write_run_records_csv,
    write_table_summary,
)
from f2.suites.common.w2v.artifacts import load_lookup_artifact
from f2.suites.common.w2v.evaluation import evaluate_analogies, parse_analogy_questions
from repro_core.context import RuntimePaths

from .. import paper

_OBJECTIVES = (
    "neg5-no-subsampling",
    "neg5-subsampling",
    "neg15-no-subsampling",
    "neg15-subsampling",
    "hs-no-subsampling",
    "hs-subsampling",
)


@dataclass(frozen=True)
class Table3RunResult:
    condition: str
    seed: int
    mlflow_run_id: str
    phrase_accuracy_percent: float
    observed_training_seconds: float
    included_questions: int
    total_questions: int


def _is_table3_run(record: dict[str, Any], corpus_source: str) -> bool:
    tags = record.get("tags") or {}
    spec = tags.get("experiment_spec.id")
    study = tags.get("study") or tags.get("f2.experiment_id")
    if spec != "w2v2-phrase-skipgram-1b-objectives" and study not in ("table3", "e02"):
        return False
    variant = tags.get("implementation.variant") or tags.get("variant") or ""
    corpus_id = tags.get("corpus.id") or ""
    if corpus_id:
        return corpus_id == corpus_source
    if "--" in variant:
        return variant.split("--")[0] == corpus_source
    return False


def _extract_objective(record: dict[str, Any]) -> str | None:
    tags = record.get("tags") or {}
    cond = tags.get("condition") or tags.get("variant") or ""
    for obj in _OBJECTIVES:
        if obj in cond:
            return obj
    return None


def analyze_table3(
    runs: list[Any],
    phrase_questions_path: Path | None,
    *,
    output_dir: Path,
    corpus_source: str = "news-1b",
    paths: RuntimePaths | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = paths or RuntimePaths.from_environment()

    if phrase_questions_path is None or not phrase_questions_path.is_file():
        return unavailable_report(
            output_dir,
            "Phrase analogy benchmark questions-phrases.txt is missing.",
        )

    questions_bytes = phrase_questions_path.read_bytes()
    questions = parse_analogy_questions(questions_bytes.splitlines())
    questions_sha256 = hashlib.sha256(questions_bytes).hexdigest()

    cache_dir = paths.analysis_cache_root / "f2/w2v2/analysis/table3"
    eval_cache = W2VTableEvaluationCache(
        cache_dir,
        protocol="w2v2-table3",
        result_cls=dict,
        questions_sha256=questions_sha256,
    )

    matching_runs = [r for r in runs if _is_table3_run(r, corpus_source)]
    if not matching_runs:
        return unavailable_report(
            output_dir,
            f"No matching completed W2V2 Table 3 runs found for corpus {corpus_source}.",
        )

    run_results: list[Table3RunResult] = []
    for r in matching_runs:
        run_id = r.get("run_id")
        cond = _extract_objective(r)
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
                    "phrase_accuracy_percent": 100.0 * eval_res.overall.require_score(),
                    "included_questions": eval_res.overall.valid_count,
                    "total_questions": eval_res.overall.total_count,
                }
                eval_cache.store(run_id, res_dict)
            except Exception:
                continue

        run_results.append(
            Table3RunResult(
                condition=cond,
                seed=seed,
                mlflow_run_id=run_id,
                phrase_accuracy_percent=res_dict["phrase_accuracy_percent"],
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

    by_condition: dict[str, list[Table3RunResult]] = {}
    for res in run_results:
        by_condition.setdefault(res.condition, []).append(res)

    headers = [
        "Objective / Subsampling",
        "Phrase Accuracy % (Mean ± SD)",
        "Paper Reference (%)",
    ]
    rows = []
    for cond in _OBJECTIVES:
        items = by_condition.get(cond, [])
        ref = paper.TABLE3.get(cond, "-")
        ref_str = f"{ref}%" if ref != "-" else "-"
        if not items:
            rows.append([cond, "N/A", ref_str])
            continue
        acc_str = format_mean_sd([it.phrase_accuracy_percent for it in items])
        rows.append([cond, acc_str, ref_str])

    write_table_summary(
        output_dir,
        title="W2V2 Table 3: Phrase-Analogy Accuracy on News-1B",
        paper_reference=paper.SOURCE,
        headers=headers,
        rows=rows,
    )
    records_rows = [
        {
            "condition": r.condition,
            "seed": r.seed,
            "mlflow_run_id": r.mlflow_run_id,
            "phrase_accuracy_percent": r.phrase_accuracy_percent,
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
                "table": "3",
                "corpus": corpus_source,
                "results": records_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return output_dir


def analyze_table3_sources(
    cache_root: Path,
    phrase_questions_path: Path | None,
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
        out = paths.analysis_results_root / f"f2/w2v2/table03/{c_source}"
        results.append(
            analyze_table3(
                runs,
                phrase_questions_path,
                output_dir=out,
                corpus_source=c_source,
                paths=paths,
            )
        )
    return results


__all__ = ["Table3RunResult", "analyze_table3", "analyze_table3_sources"]
