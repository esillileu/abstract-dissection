"""Table 2 word-analogy evaluation over durable W2V1 MLflow artifacts."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from f2.storage.analysis_cache import LocalResults, unavailable_report
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

CORPUS_SOURCES = ("wmt", "lm1b", "umbc")
_DIMENSIONS = (50, 100, 300, 600)
_TRAINING_WORDS = (24, 49, 98, 196, 391, 783)
_SEEDS = (1, 7, 19)
_VOCABULARY_LIMIT = 30_000
_EVALUATION_PROTOCOL = "compute-accuracy-v1"


@dataclass(frozen=True)
class Table2RunResult:
    training_words_millions: int
    vector_dimension: int
    seed: int
    mlflow_run_id: str
    total_accuracy_percent: float
    semantic_accuracy_percent: float
    syntactic_accuracy_percent: float
    observed_training_seconds: float
    included_questions: int
    total_questions: int


def _complete_conditions(
    runs: list[Any], corpus_source: str
) -> dict[tuple[int, int], dict[int, Any]]:
    grouped: dict[tuple[int, int], dict[int, Any]] = {}
    for run in runs:
        tags = run.data.tags
        spec = tags.get("experiment_spec.id") or tags.get("repro.experiment_spec_id")
        if spec != "w2v1-table2-cbow":
            continue
        variant = tags.get("implementation.variant") or ""
        if not variant.startswith(f"{corpus_source}--"):
            continue
        try:
            seed = int(tags.get("seed") or tags.get("f2.seed", ""))
        except ValueError:
            continue
        if seed not in _SEEDS:
            continue
        # Extract dimension and training words from variant or tags
        # variant format: <corpus>--d<dim>-w<words>m
        cond = variant.split("--", 1)[1]
        parts = cond.split("-")
        try:
            if (
                len(parts) != 2
                or not parts[0].startswith("d")
                or not (parts[1].startswith("w") and parts[1].endswith("m"))
            ):
                continue
            dim = int(parts[0][1:])
            words = int(parts[1][1:-1])
        except ValueError:
            continue
        if dim not in _DIMENSIONS or words not in _TRAINING_WORDS:
            continue
        key = (words, dim)
        seeds = grouped.setdefault(key, {})
        if seed not in seeds:
            seeds[seed] = run
    return {key: seeds for key, seeds in grouped.items() if set(seeds) == set(_SEEDS)}


def analyze_table2(
    cache_root: Path,
    questions_path: Path,
    *,
    corpus_source: str = "wmt",
    paths: RuntimePaths | None = None,
) -> Path:
    paths = paths or RuntimePaths.from_environment()
    if not Path(questions_path).is_file():
        return unavailable_report(
            paths.analysis_results_root / f"f2/w2v1/table02/{corpus_source}",
            "cached evaluation questions missing",
        )
    output = paths.analysis_results_root / f"f2/w2v1/table02/{corpus_source}"
    output.mkdir(parents=True, exist_ok=True)
    cache = LocalResults(cache_root)
    runs = cache.runs()
    selected = _complete_conditions(runs, corpus_source)
    if not selected:
        return unavailable_report(output, "no complete condition sets found")

    questions_bytes = Path(questions_path).read_bytes()
    questions_sha256 = hashlib.sha256(questions_bytes).hexdigest()
    questions = parse_analogy_questions(questions_bytes.splitlines())
    evaluation_cache = W2VTableEvaluationCache(
        paths.analysis_cache_root / "f2/w2v1/analysis/table2",
        protocol=_EVALUATION_PROTOCOL,
        result_cls=Table2RunResult,
        questions_sha256=questions_sha256,
        extra_signature={"vocabulary_limit": _VOCABULARY_LIMIT},
    )

    records: list[Table2RunResult] = []
    unavailable: list[dict[str, Any]] = []
    cache_hits = cache_misses = 0

    for condition, seed_runs in sorted(selected.items()):
        for seed in _SEEDS:
            run = seed_runs[seed]
            cached = evaluation_cache.load(run.info.run_id)
            if cached is not None:
                records.append(cached)
                cache_hits += 1
                continue
            try:
                cache_misses += 1
                lookup = load_lookup_artifact(cache.get(run.info.run_id, "lookup"))
                evaluation = evaluate_analogies(
                    lookup,
                    questions,
                    vocabulary_limit=_VOCABULARY_LIMIT,
                    batch_size=256,
                )
                seconds = observed_training_seconds(
                    cache.get(run.info.run_id, "metrics/observations.csv")
                )
                record = Table2RunResult(
                    training_words_millions=condition[0],
                    vector_dimension=condition[1],
                    seed=seed,
                    mlflow_run_id=run.info.run_id,
                    total_accuracy_percent=100.0 * evaluation.overall.require_score(),
                    semantic_accuracy_percent=100.0
                    * evaluation.semantic.require_score(),
                    syntactic_accuracy_percent=100.0
                    * evaluation.syntactic.require_score(),
                    observed_training_seconds=seconds,
                    included_questions=evaluation.overall.valid_count,
                    total_questions=evaluation.overall.total_count,
                )
                evaluation_cache.store(run.info.run_id, record)
                records.append(record)
            except (FileNotFoundError, KeyError, ValueError) as exc:
                unavailable.append(
                    {
                        "condition": f"d{condition[1]}-w{condition[0]}m",
                        "seed": seed,
                        "run_id": run.info.run_id,
                        "status": "no_data",
                        "reason": str(exc),
                    }
                )

    print(
        f"W2V1 Table 2 analogy cache: {cache_hits} hit(s), {cache_misses} miss(es)",
        file=sys.stderr,
    )
    (output / "unavailable.json").write_text(
        json.dumps(unavailable, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_run_records_csv(output / "table2-runs.csv", records)

    # Render standardized Markdown Table (Accuracy & Time Mean ± SD)
    grouped: dict[tuple[int, int], list[Table2RunResult]] = {}
    for r in records:
        grouped.setdefault((r.training_words_millions, r.vector_dimension), []).append(
            r
        )

    rows: list[list[str]] = []
    for (words, dim), items in sorted(grouped.items()):
        acc_str, time_str = format_accuracy_and_time(
            items, "total_accuracy_percent", precision=1
        )
        sem_str = format_mean_sd(
            [x.semantic_accuracy_percent for x in items], precision=1
        )
        syn_str = format_mean_sd(
            [x.syntactic_accuracy_percent for x in items], precision=1
        )
        rows.append(
            [
                f"{words}M",
                str(dim),
                str(len(items)),
                acc_str,
                sem_str,
                syn_str,
                time_str,
            ]
        )

    write_table_summary(
        output / "summary.md",
        title=f"W2V1 Table 2 — {corpus_source.upper()} (Word Analogy Accuracy & Training Time)",
        headers=[
            "Words",
            "Dim",
            "Runs",
            "Total Accuracy (%)",
            "Semantic (%)",
            "Syntactic (%)",
            "Training Time",
        ],
        rows=rows,
    )

    (output / "results.json").write_text(
        json.dumps(
            {
                "protocol": {
                    "evaluation_protocol": _EVALUATION_PROTOCOL,
                    "vocabulary_limit": _VOCABULARY_LIMIT,
                    "questions_sha256": questions_sha256,
                },
                "runs": [asdict(record) for record in records],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return output


def analyze_table2_sources(
    cache_root: Path,
    questions_path: Path,
    *,
    corpus_source: str | None = None,
    paths: RuntimePaths | None = None,
) -> tuple[Path, ...]:
    sources = CORPUS_SOURCES if corpus_source is None else (corpus_source,)
    return tuple(
        analyze_table2(
            cache_root,
            questions_path,
            corpus_source=source,
            paths=paths,
        )
        for source in sources
    )


__all__ = [
    "CORPUS_SOURCES",
    "Table2RunResult",
    "analyze_table2",
    "analyze_table2_sources",
    "observed_training_seconds",
]
