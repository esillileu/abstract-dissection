"""Markdown and statistical reporting helpers for Table 5."""

from __future__ import annotations

import statistics
from pathlib import Path

from .table05_types import TABLE5_SPEC, ConditionKey, Table5RunResult


def fmt_stats(records: list[Table5RunResult], field: str) -> str:
    if not records:
        return "—"
    nums = [float(getattr(r, field)) for r in records]
    if len(nums) == 1:
        return f"{nums[0]:.2f}"
    return f"{statistics.fmean(nums):.2f} ± {statistics.stdev(nums):.2f}"


def get_mean(records: list[Table5RunResult], field: str) -> float | None:
    if not records:
        return None
    nums = [float(getattr(r, field)) for r in records]
    return float(statistics.fmean(nums))


def write_table5_summary(
    path: Path, corpus: str, records: list[Table5RunResult], digest: str
) -> None:
    lines = [
        f"# W2V1 Table 5 — {corpus.upper()}",
        "",
        "## Table 5 Reproduction Comparison",
        "",
        "| Model | Dim | Words | Epochs | Original Semantic | Repro Semantic (%) | Original Syntactic | Repro Syntactic (%) | Original Total | Repro Total (%) | Original Time | Repro Time (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    grouped: dict[ConditionKey, list[Table5RunResult]] = {}
    for r in records:
        key = (r.architecture, r.dimension, r.training_words_millions, r.epochs)
        grouped.setdefault(key, []).append(r)

    for (
        arch,
        dim,
        words,
        epochs,
        orig_sem,
        orig_syn,
        orig_tot,
        orig_time,
    ) in TABLE5_SPEC:
        cond_records = grouped.get((arch, dim, words, epochs), [])
        repro_sem = fmt_stats(cond_records, "semantic_accuracy_percent")
        repro_syn = fmt_stats(cond_records, "syntactic_accuracy_percent")
        repro_tot = fmt_stats(cond_records, "total_accuracy_percent")
        repro_sec = fmt_stats(cond_records, "observed_training_seconds")
        model_name = "CBOW" if arch == "cbow" else "Skip-gram"
        words_label = f"{words}M" if words < 1000 else f"{words / 1000:.1f}B"

        lines.append(
            f"| {model_name} | {dim} | {words_label} | {epochs} | "
            f"{orig_sem:.1f} | {repro_sem} | {orig_syn:.1f} | {repro_syn} | "
            f"{orig_tot:.1f} | {repro_tot} | {orig_time} | {repro_sec} |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## Core Comparisons",
            "",
            "### A. Repeated Training (783M x 3 epochs) vs More Data (1.6B x 1 epoch)",
            "",
            "| Model | Dim | Condition 1 | Condition 2 | Δ Semantic | Δ Syntactic | Δ Total | Runtime Ratio (1.6B 1ep / 783M 3ep) |",
            "|---|---:|---|---|---:|---:|---:|---:|",
        ]
    )

    for arch in ("cbow", "skipgram"):
        r_3ep = grouped.get((arch, 300, 783, 3), [])
        r_1ep = grouped.get((arch, 300, 1600, 1), [])
        model_name = "CBOW" if arch == "cbow" else "Skip-gram"
        if r_3ep and r_1ep:
            d_sem = get_mean(r_1ep, "semantic_accuracy_percent") - get_mean(
                r_3ep, "semantic_accuracy_percent"
            )
            d_syn = get_mean(r_1ep, "syntactic_accuracy_percent") - get_mean(
                r_3ep, "syntactic_accuracy_percent"
            )
            d_tot = get_mean(r_1ep, "total_accuracy_percent") - get_mean(
                r_3ep, "total_accuracy_percent"
            )
            ratio = get_mean(r_1ep, "observed_training_seconds") / max(
                1e-6, get_mean(r_3ep, "observed_training_seconds")
            )
            lines.append(
                f"| {model_name} | 300 | 783M x 3ep | 1.6B x 1ep | {d_sem:+.2f}% | {d_syn:+.2f}% | {d_tot:+.2f}% | {ratio:.2f}x |"
            )
        else:
            lines.append(
                f"| {model_name} | 300 | 783M x 3ep | 1.6B x 1ep | — | — | — | — |"
            )

    lines.extend(
        [
            "",
            "### B. Epoch Effect (783M x 3 epochs vs 783M x 1 epoch)",
            "",
            "| Model | Dim | Words | 3 Epochs Total (%) | 1 Epoch Total (%) | Δ Total (3ep - 1ep) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )

    for arch in ("cbow", "skipgram"):
        r_3ep = grouped.get((arch, 300, 783, 3), [])
        r_1ep = grouped.get((arch, 300, 783, 1), [])
        model_name = "CBOW" if arch == "cbow" else "Skip-gram"
        if r_3ep and r_1ep:
            m_3ep = get_mean(r_3ep, "total_accuracy_percent")
            m_1ep = get_mean(r_1ep, "total_accuracy_percent")
            lines.append(
                f"| {model_name} | 300 | 783M | {m_3ep:.2f}% | {m_1ep:.2f}% | {m_3ep - m_1ep:+.2f}% |"
            )
        else:
            lines.append(f"| {model_name} | 300 | 783M | — | — | — |")

    lines.extend(
        [
            "",
            "### C. Vector Dimension Effect (300d vs 600d at 783M x 1 epoch)",
            "",
            "| Model | Words | Epochs | 300d Total (%) | 600d Total (%) | Δ Total (600d - 300d) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )

    for arch in ("cbow", "skipgram"):
        r_300d = grouped.get((arch, 300, 783, 1), [])
        r_600d = grouped.get((arch, 600, 783, 1), [])
        model_name = "CBOW" if arch == "cbow" else "Skip-gram"
        if r_300d and r_600d:
            m_300d = get_mean(r_300d, "total_accuracy_percent")
            m_600d = get_mean(r_600d, "total_accuracy_percent")
            lines.append(
                f"| {model_name} | 783M | 1 | {m_300d:.2f}% | {m_600d:.2f}% | {m_600d - m_300d:+.2f}% |"
            )
        else:
            lines.append(f"| {model_name} | 783M | 1 | — | — | — |")

    lines.extend(
        [
            "",
            "---",
            "",
            "### Notes",
            "- 1M lexical vocabulary cap is a reconstruction decision matching Table 4.",
            "- Evaluation uses the full saved lookup vocabulary. OOV questions are excluded.",
            f"- questions-words.txt SHA-256: `{digest}`.",
            "- Observed time sums final dense elapsed observations per epoch.",
        ]
    )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


__all__ = ["fmt_stats", "get_mean", "write_table5_summary"]
