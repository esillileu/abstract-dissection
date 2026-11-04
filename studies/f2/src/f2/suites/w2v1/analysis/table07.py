"""Microsoft Research Sentence Completion Challenge Table 7 reproduction and analysis."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
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
from f2.suites.common.w2v.artifacts import load_lookup_artifact
from f2.suites.common.w2v.evaluation import (
    evaluate_msr_sentence_completion,
    parse_msr_sentence_completion_questions,
)
from repro_core.context import RuntimePaths

from ..resources import MSR_ANSWERS_SHA256, MSR_QUESTIONS_SHA256

TABLE7_SPEC = ("skipgram", 640, 50, 1, 48.0)
_SEEDS = {1, 7, 19}
_EVALUATION_PROTOCOL = "table7-v1"


@dataclass(frozen=True)
class Table7RunResult:
    architecture: str
    dimension: int
    training_words_millions: int
    epochs: int
    corpus: str
    seed: int
    mlflow_run_id: str
    overall_accuracy_percent: float
    dev_accuracy_percent: float
    test_accuracy_percent: float
    total_questions: int
    observed_training_seconds: float


def collect_table7_runs(runs: list[Any]) -> dict[int, Any]:
    grouped: dict[int, Any] = {}
    for run in runs:
        tags = run.data.tags
        variant = tags.get("implementation.variant", "")
        if variant != "holmes--skipgram-d640-w50m":
            continue
        try:
            seed = int(tags.get("seed", tags.get("f2.seed", "")))
        except ValueError:
            continue
        if seed in _SEEDS:
            grouped[seed] = run
    return grouped


def analyze_table7_sources(
    cache_root: Path,
    questions_path: Path | None = None,
    answers_path: Path | None = None,
    *,
    corpus_source: str | None = None,
    paths: RuntimePaths | None = None,
) -> tuple[Path, ...]:
    if corpus_source not in (None, "holmes"):
        raise ValueError("Table 7 requires Holmes corpus; --corpus must be 'holmes'")
    paths = paths or RuntimePaths.from_environment()

    if questions_path is None:
        benchmarks = paths.inputs("f2", "benchmarks")
        default_q = benchmarks / "Holmes.lm_format.questions.txt"
        sub_q = benchmarks / "msr_sentence_completion/Holmes.lm_format.questions.txt"
        if default_q.is_file():
            questions_path = default_q
        elif sub_q.is_file():
            questions_path = sub_q
        else:
            questions_path = (
                cache_root
                / "evaluation/msr_sentence_completion/Holmes.lm_format.questions.txt"
            )

    if answers_path is None:
        benchmarks = paths.inputs("f2", "benchmarks")
        default_a = benchmarks / "Holmes.lm_format.answers.txt"
        sub_a = benchmarks / "msr_sentence_completion/Holmes.lm_format.answers.txt"
        if default_a.is_file():
            answers_path = default_a
        elif sub_a.is_file():
            answers_path = sub_a
        else:
            answers_path = (
                cache_root
                / "evaluation/msr_sentence_completion/Holmes.lm_format.answers.txt"
            )

    for path, name in ((questions_path, "questions"), (answers_path, "answers")):
        if path is None or not Path(path).is_file():
            out = unavailable_report(
                paths.analysis_results_root / "f2/w2v1/table07",
                f"cached MSR {name} missing",
            )
            return (out,)

    q_bytes = Path(questions_path).read_bytes()
    a_bytes = Path(answers_path).read_bytes()
    q_digest = hashlib.sha256(q_bytes).hexdigest()
    a_digest = hashlib.sha256(a_bytes).hexdigest()

    parsed_questions = parse_msr_sentence_completion_questions(
        q_bytes.splitlines(),
        a_bytes.splitlines(),
    )

    output = paths.analysis_results_root / "f2/w2v1/table07/holmes"
    output.mkdir(parents=True, exist_ok=True)
    cache = LocalResults(cache_root)
    runs = cache.runs()
    selected_seeds = collect_table7_runs(runs)
    if set(selected_seeds.keys()) != _SEEDS:
        return (unavailable_report(output, "no complete condition sets found"),)

    evaluation_cache = W2VTableEvaluationCache(
        paths.analysis_cache_root / "f2/w2v1/analysis/table7",
        protocol=_EVALUATION_PROTOCOL,
        result_cls=Table7RunResult,
        questions_sha256=q_digest,
        extra_signature={"answers_sha256": a_digest},
    )

    records: list[Table7RunResult] = []
    unavailable: list[dict[str, Any]] = []
    cache_hits = cache_misses = 0

    for seed in sorted(_SEEDS):
        run = selected_seeds[seed]
        cached = evaluation_cache.load(run.info.run_id)
        if cached is not None:
            records.append(cached)
            cache_hits += 1
            continue
        try:
            cache_misses += 1
            lookup = load_lookup_artifact(cache.get(run.info.run_id, "lookup"))
            eval_res = evaluate_msr_sentence_completion(lookup, parsed_questions)
            seconds = observed_training_seconds(
                cache.get(run.info.run_id, "metrics/observations.csv")
            )
            record = Table7RunResult(
                architecture="skipgram",
                dimension=640,
                training_words_millions=50,
                epochs=1,
                corpus="holmes",
                seed=seed,
                mlflow_run_id=run.info.run_id,
                overall_accuracy_percent=100.0 * eval_res.overall.require_score(),
                dev_accuracy_percent=100.0 * eval_res.dev.require_score(),
                test_accuracy_percent=100.0 * eval_res.test.require_score(),
                total_questions=eval_res.overall.total_count,
                observed_training_seconds=seconds,
            )
            evaluation_cache.store(run.info.run_id, record)
            records.append(record)
        except (FileNotFoundError, KeyError, ValueError) as exc:
            unavailable.append(
                {
                    "condition": "skipgram-d640-w50m",
                    "seed": seed,
                    "run_id": run.info.run_id,
                    "reason": str(exc),
                }
            )

    print(
        f"W2V1 Table 7 cache: {cache_hits} hit(s), {cache_misses} miss(es)",
        file=sys.stderr,
    )
    (output / "unavailable.json").write_text(
        json.dumps(unavailable, indent=2) + "\n", encoding="utf-8"
    )
    write_run_records_csv(output / "runs.csv", records)

    tot_str, time_str = format_accuracy_and_time(records, "overall_accuracy_percent")
    dev_str = format_mean_sd([x.dev_accuracy_percent for x in records])
    test_str = format_mean_sd([x.test_accuracy_percent for x in records])

    write_table_summary(
        output / "summary.md",
        title="W2V1 Table 7 — MSR Sentence Completion Challenge",
        headers=[
            "Model",
            "Dim",
            "Words",
            "Epochs",
            "Runs",
            "Total (%)",
            "Dev (%)",
            "Test (%)",
            "Training Time",
            "Paper Target (%)",
        ],
        rows=[
            [
                "skipgram",
                "640",
                "50M",
                "1",
                str(len(records)),
                tot_str,
                dev_str,
                test_str,
                time_str,
                "48.0",
            ]
        ],
    )
    return (output,)


__all__ = [
    "MSR_ANSWERS_SHA256",
    "MSR_QUESTIONS_SHA256",
    "TABLE7_SPEC",
    "Table7RunResult",
    "analyze_table7_sources",
    "collect_table7_runs",
]
