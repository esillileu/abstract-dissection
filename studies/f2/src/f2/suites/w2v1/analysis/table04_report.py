"""Reporting and formatting routines for Table 4."""

from __future__ import annotations

import statistics
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .table04 import Table4RunResult


def write_summary(
    path: Path,
    corpus: str,
    records: list[Table4RunResult],
    digest: str,
    reference: dict[str, tuple[float, float, float]],
    external_reference: tuple[tuple[str, int, int, float, float, float], ...],
) -> None:
    lines = [
        f"# W2V1 Table 4 — {corpus.upper()}",
        "",
        "Only conditions with all three durable canonical runs are shown.",
        "",
        "| Architecture | Corpus | Runs | Semantic mean ± sample SD (%) | Syntactic mean ± sample SD (%) | Total mean ± sample SD (%) | Questions included / total | Observed training time mean ± sample SD (s) | MLflow run IDs | Paper reference: semantic / syntactic / total (%) |",
        "|---|---|---:|---:|---:|---:|---|---:|---|---|",
    ]
    for architecture in reference:
        values = [record for record in records if record.architecture == architecture]
        if not values:
            if architecture.startswith("nnlm-") and corpus == "fineweb":
                ref_str = " / ".join(
                    f"{number:.1f}" for number in reference[architecture]
                )
                lines.append(
                    f"| {architecture} | {corpus} | 0 | unavailable | unavailable | unavailable | — | — | awaiting canonical runs | {ref_str} |"
                )
            continue
        coverage = ", ".join(
            f"{x.included_questions}/{x.total_questions}" for x in values
        )
        ref_str = " / ".join(f"{number:.1f}" for number in reference[architecture])
        lines.append(
            f"| {architecture} | {corpus} | {len(values)} | "
            f"{_summary(values, 'semantic_accuracy_percent')} | {_summary(values, 'syntactic_accuracy_percent')} | "
            f"{_summary(values, 'total_accuracy_percent')} | {coverage} | "
            f"{_summary(values, 'observed_training_seconds')} | "
            f"{', '.join(x.mlflow_run_id for x in values)} | {ref_str} |"
        )
    lines.extend(
        [
            "",
            "External published vectors — paper reference only; no reproduction runs:",
            "",
            "| Model | Dimensions | Words | Semantic | Syntactic | Total (%) |",
            "|---|---:|---|---:|---:|---:|",
        ]
    )
    lines.extend(
        f"| {name} | {dim} | {words} | {sem} | {synt} | {total} |"
        for name, dim, words, sem, synt, total in external_reference
    )
    lines.extend(
        [
            "",
            "Our NNLM 20/50/100d uses the existing FineWeb 2013 6B surrogate policy. History=8/hidden=640/tanh and concrete PS/AdaGrad schedule are reconstruction decisions, not Table 4/6 specifications.",
            "Table 6 reuses the canonical Our NNLM 100d run IDs; it adds no NNLM training run.",
            "Paper reference: https://arxiv.org/pdf/1301.3781 (Tables 4 and 6).",
            "The 1M lexical vocabulary cap is a reconstruction decision; Table 4 does not state it explicitly.",
            "Evaluation uses the full saved lookup vocabulary. OOV questions are excluded.",
            "Scores preserve the existing compute-accuracy casing, cosine normalization, positive-score, and source-exclusion behavior.",
            f"questions-words.txt SHA-256: `{digest}`.",
            "Observed time sums final dense elapsed observations per epoch and may omit each epoch's tail.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _summary(values: list[Table4RunResult], field: str) -> str:
    numbers = [float(getattr(value, field)) for value in values]
    return f"{statistics.fmean(numbers):.2f} ± {statistics.stdev(numbers):.2f}"


__all__ = ["_summary", "write_summary"]
