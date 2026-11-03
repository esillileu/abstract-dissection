"""Reporting and formatting routines for Table 7."""

from __future__ import annotations

import csv
import statistics
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .table07 import Table7RunResult

HOLMES_CORPUS_SHA256 = "d41d8cd98f00b204e9800998ecf8427e"
HOLMES_RAW_BYTES = 50_000_000
MSR_QUESTIONS_SHA256 = (
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
)
MSR_ANSWERS_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def write_table7_markdown(path: Path, results: list[Table7RunResult]) -> None:
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


def write_table7_csv(path: Path, results: list[Table7RunResult]) -> None:
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
    "write_table7_csv",
    "write_table7_markdown",
]
