"""Microsoft Research Sentence Completion Challenge Table 7 reproduction and analysis."""

from __future__ import annotations

import csv
import json
import re
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from f2.suites.w2v.artifacts import load_lookup_artifact
from f2.suites.w2v.cache import LocalResults, unavailable_report
from f2.suites.w2v.evaluation import (
    evaluate_msr_sentence_completion,
    parse_msr_sentence_completion_questions,
)
from repro_core.context import RuntimePaths

from .analysis import observed_training_seconds
from .resources import (
    HOLMES_CORPUS_SHA256,
    HOLMES_RAW_BYTES,
    MSR_ANSWERS_SHA256,
    MSR_QUESTIONS_SHA256,
)

# Condition specification for Table 7: Skip-gram 640d on 50M words Holmes corpus
# (model, dim, words_millions, epochs, orig_accuracy)
TABLE7_SPEC = ("skipgram", 640, 50, 1, 48.0)

_SEEDS = {1, 7, 19}

_T7_SLOT_PATTERN = re.compile(r"w2v1-table7-holmes-r1-skipgram-d640-w50m-s(\d+)")

# Canonical benchmark mirror URLs and SHA-256 checksums


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


_EVALUATION_PROTOCOL = "table7-v1"


class Table7EvaluationCache:
    """Signature-checked per-run cache for reconstructible Table 7 evaluation results."""

    def __init__(
        self, root: Path, *, questions_sha256: str, answers_sha256: str
    ) -> None:
        self.root = root
        self.signature = {
            "schema_version": 2,
            "evaluation_protocol": _EVALUATION_PROTOCOL,
            "questions_sha256": questions_sha256,
            "answers_sha256": answers_sha256,
        }

    def load(self, run_id: str) -> Table7RunResult | None:
        path = self._path(run_id)
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("signature") != self.signature:
                return None
            result = Table7RunResult(**payload["result"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None
        return result if result.mlflow_run_id == run_id else None

    def store(self, result: Table7RunResult) -> None:
        path = self._path(result.mlflow_run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "signature": self.signature,
                    "result": asdict(result),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    def _path(self, run_id: str) -> Path:
        if re.fullmatch(r"[0-9a-zA-Z_-]+", run_id) is None:
            raise ValueError(f"invalid MLflow run ID: {run_id}")
        return self.root / f"{run_id}.json"


def collect_table7_runs(runs: list[Any]) -> dict[int, Any]:
    """Group durable Table 7 runs by seed -> run."""
    grouped: dict[int, Any] = {}
    for run in runs:
        slot = run.data.tags.get("f2.planned_run_slot_id", "")
        match = _T7_SLOT_PATTERN.fullmatch(slot)
        if match:
            seed = int(match.group(1))
            if seed in _SEEDS:
                grouped[seed] = run
    return grouped


def analyze_table7_sources(
    cache_root: Path,
    questions_path: Path | None = None,
    *,
    paths: RuntimePaths | None = None,
) -> list[Path]:
    """Aggregate durable Table 7 runs, evaluate sentence completion, and write reports."""
    paths = paths or RuntimePaths.from_environment()

    artifact_cache = LocalResults(cache_root)
    runs = artifact_cache.runs()

    benchmark = (
        Path(questions_path) if questions_path else Path(cache_root) / "evaluation"
    )
    base = benchmark.parent if benchmark.suffix else benchmark
    q_file = base / "Holmes.lm_format.questions.txt"
    a_file = base / "Holmes.lm_format.answers.txt"

    if not q_file.is_file() or not a_file.is_file():
        output = unavailable_report(
            paths.cache_root / "f2/w2v1/analysis/table7",
            "cached MSR questions or answers missing",
        )
        return [output / "summary.md", output / "unavailable.json"]
    questions = parse_msr_sentence_completion_questions(
        q_file.read_bytes().splitlines(),
        a_file.read_bytes().splitlines(),
    )

    out_dir = paths.cache_root / "f2/w2v1/analysis/table7"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_md = out_dir / "summary.md"
    summary_csv = out_dir / "summary.csv"

    evaluation_cache = Table7EvaluationCache(
        paths.cache_root / "f2/w2v1/table7",
        questions_sha256=MSR_QUESTIONS_SHA256,
        answers_sha256=MSR_ANSWERS_SHA256,
    )

    grouped_runs = collect_table7_runs(runs)

    run_results: list[Table7RunResult] = []
    unavailable = []
    cache_hits = 0
    cache_misses = 0
    for seed, run in sorted(grouped_runs.items()):
        cached = evaluation_cache.load(run.info.run_id)
        if cached is not None and cached.seed == seed:
            run_results.append(cached)
            cache_hits += 1
            continue
        try:
            cache_misses += 1
            lookup_dir = artifact_cache.get(run.info.run_id, "lookup")
            lookup = load_lookup_artifact(lookup_dir)
            evaluation = evaluate_msr_sentence_completion(lookup, questions)
            metrics_file = artifact_cache.get(
                run.info.run_id, "metrics/observations.csv"
            )
            seconds = observed_training_seconds(metrics_file)

            record = Table7RunResult(
                architecture="skipgram",
                dimension=640,
                training_words_millions=50,
                epochs=1,
                corpus="holmes",
                seed=seed,
                mlflow_run_id=run.info.run_id,
                overall_accuracy_percent=round(
                    evaluation.overall.require_score() * 100.0, 2
                ),
                dev_accuracy_percent=round(evaluation.dev.require_score() * 100.0, 2),
                test_accuracy_percent=round(evaluation.test.require_score() * 100.0, 2),
                total_questions=len(questions),
                observed_training_seconds=seconds,
            )
            evaluation_cache.store(record)
            run_results.append(record)

        except (FileNotFoundError, KeyError, ValueError) as exc:
            unavailable.append(
                {"run_id": run.info.run_id, "status": "no_data", "reason": str(exc)}
            )
    (out_dir / "unavailable.json").write_text(json.dumps(unavailable, indent=2) + "\n")
    print(
        f"W2V1 Table 7 evaluation cache: {cache_hits} hit(s), {cache_misses} miss(es)",
        file=sys.stderr,
    )
    _write_table7_markdown(summary_md, run_results)
    _write_table7_csv(summary_csv, run_results)

    return [summary_md, summary_csv]


def _write_table7_markdown(path: Path, results: list[Table7RunResult]) -> None:
    lines = [
        "# Table 7: Microsoft Research Sentence Completion Challenge Reproduction",
        "",
        "Evaluation of 640-dimensional Skip-gram on 1,040 questions from the MSR Sentence Completion Challenge.",
        "",
        "| Architecture | Dimension | Training Words | Epochs | Corpus | Paper Acc [%] | Repro Mean [%] | Repro Std [%] | Seeds | Questions |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    if results:
        overall_scores = [r.overall_accuracy_percent for r in results]
        mean_acc = statistics.mean(overall_scores)
        std_acc = statistics.stdev(overall_scores) if len(overall_scores) > 1 else 0.0
        seed_str = ",".join(str(r.seed) for r in results)
        lines.append(
            f"| Skip-gram | 640 | 50M | 1 | Holmes | 48.0 | {mean_acc:.2f} | {std_acc:.2f} | {seed_str} | {results[0].total_questions} |"
        )
        lines.extend(
            [
                "",
                "## Seed-level Results",
                "",
                "| Seed | Overall Acc [%] | Dev Acc (520) [%] | Test Acc (520) [%] | Training Time [s] | MLflow Run ID |",
                "| :--- | :--- | :--- | :--- | :--- | :--- |",
            ]
        )
        for r in results:
            lines.append(
                f"| {r.seed} | {r.overall_accuracy_percent:.2f} | {r.dev_accuracy_percent:.2f} | {r.test_accuracy_percent:.2f} | {r.observed_training_seconds:.1f} | `{r.mlflow_run_id[:8]}` |"
            )
    else:
        lines.append(
            "| Skip-gram | 640 | 50M | 1 | Holmes | 48.0 | Pending | Pending | - | 1,040 |"
        )

    lines.extend(
        [
            "",
            "## Provenance & Data Specifications",
            "",
            f"- Training Corpus: Project Gutenberg Holmes collection (522 books, {HOLMES_RAW_BYTES:,} bytes, concatenated SHA-256 `{HOLMES_CORPUS_SHA256}`).",
            f"- Evaluation Questions: 1,040 SAT-style sentence completion questions (SHA-256 `{MSR_QUESTIONS_SHA256}`).",
            f"- Evaluation Gold Answers: 1,040 gold completion sentences (SHA-256 `{MSR_ANSWERS_SHA256}`).",
            "- Sentence Scoring: Candidate missing word at input, sum of predictions for all surrounding context words (cosine similarity on L2-normalized embeddings).",
            "",
        ]
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def _write_table7_csv(path: Path, results: list[Table7RunResult]) -> None:
    fieldnames = [
        "architecture",
        "dimension",
        "training_words_millions",
        "epochs",
        "corpus",
        "seed",
        "mlflow_run_id",
        "overall_accuracy_percent",
        "dev_accuracy_percent",
        "test_accuracy_percent",
        "total_questions",
        "observed_training_seconds",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))


__all__ = [
    "HOLMES_CORPUS_SHA256",
    "HOLMES_RAW_BYTES",
    "MSR_ANSWERS_SHA256",
    "MSR_QUESTIONS_SHA256",
    "TABLE7_SPEC",
    "Table7RunResult",
    "analyze_table7_sources",
    "collect_table7_runs",
]
