"""Reporting and formatting routines for Table 2."""

from __future__ import annotations

import csv
import statistics
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .table02 import Table2RunResult

_DIMENSIONS = (50, 100, 300, 600)
_TRAINING_WORDS = (24, 49, 98, 196, 391, 783)
_VOCABULARY_LIMIT = 30_000


def write_run_results(path: Path, records: list[Table2RunResult]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(records[0].__dataclass_fields__.keys()) if records else [],
        )
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)


def write_summary(
    path: Path,
    records: list[Table2RunResult],
    *,
    corpus_source: str,
    questions_sha256: str,
) -> None:
    grouped: dict[tuple[int, int], list[Table2RunResult]] = {}
    for record in records:
        grouped.setdefault(
            (record.training_words_millions, record.vector_dimension), []
        ).append(record)
    lines = [
        f"# W2V1 Table 2 word analogy accuracy — {corpus_source.upper()}",
        "",
        "Cells are the mean total accuracy (%) across seeds 1, 7, and 19. "
        "Only conditions with all three durable completed runs are shown.",
        "",
        "| Training words | 50 | 100 | 300 | 600 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for words in _TRAINING_WORDS:
        cells = []
        for dimension in _DIMENSIONS:
            values = grouped.get((words, dimension))
            cells.append(
                ""
                if values is None
                else f"{statistics.fmean(x.total_accuracy_percent for x in values):.2f}"
            )
        lines.append(f"| {words}M | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "## Condition details",
            "",
            "| Training words | Dimension | Runs | Total accuracy mean ± sample SD (%) | Observed training time mean ± sample SD | Questions included / total |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for key, values in sorted(grouped.items()):
        scores = [value.total_accuracy_percent for value in values]
        training_times = [value.observed_training_seconds for value in values]
        coverage = sorted({(x.included_questions, x.total_questions) for x in values})
        coverage_text = ", ".join(f"{valid}/{total}" for valid, total in coverage)
        lines.append(
            f"| {key[0]}M | {key[1]} | {len(values)} | "
            f"{statistics.fmean(scores):.2f} ± {statistics.stdev(scores):.2f} | "
            f"{duration(statistics.fmean(training_times))} ± "
            f"{duration(statistics.stdev(training_times))} | "
            f"{coverage_text} |"
        )
    lines.extend(
        [
            "",
            "## Protocol",
            "",
            f"- Evaluation vocabulary: the first {_VOCABULARY_LIMIT:,} model vocabulary rows.",
            "- Questions with any OOV term are excluded; accuracy uses the remaining questions.",
            "- Candidate score: cosine with `b - a + c`; input terms `a`, `b`, and `c` are excluded.",
            "- Semantic is the first five source sections; syntactic is the remaining nine sections.",
            f"- `questions-words.txt` SHA-256: `{questions_sha256}`.",
            "- Per-run semantic, syntactic, total, observed training time, coverage, and MLflow IDs are in `table2-runs.csv`.",
            "- Per-run observed training time is the sum of the last dense `elapsed_seconds` observation in each epoch. It excludes setup and artifact publication, but may omit the short tail after the final observation interval.",
            "- Difference from the C executable: batched NumPy matrix multiplication replaces its scalar loop; token casing, float32 vector normalization, candidate order, positive-score initialization, filtering, and aggregation are preserved.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def duration(seconds: float) -> str:
    hours, remainder = divmod(seconds, 3600.0)
    minutes, remaining_seconds = divmod(remainder, 60.0)
    if hours >= 1.0:
        return f"{int(hours)}h {int(minutes):02d}m {remaining_seconds:04.1f}s"
    if minutes >= 1.0:
        return f"{int(minutes)}m {remaining_seconds:04.1f}s"
    return f"{remaining_seconds:.1f}s"


__all__ = ["duration", "write_run_results", "write_summary"]
