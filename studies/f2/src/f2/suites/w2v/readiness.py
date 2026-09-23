"""Canonical readiness classifications and human-readable matrix reports."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

CLASSIFICATIONS = (
    "exact_reproduction",
    "reconstruction",
    "external_baseline",
    "unsupported",
)


@dataclass(frozen=True)
class TargetDisposition:
    experiment_spec_id: str
    classification: str
    reason: str


DISPOSITIONS = (
    TargetDisposition(
        "w2v1-table2-cbow",
        "reconstruction",
        "verified WMT substitute replaces unavailable Google News bytes",
    ),
    TargetDisposition(
        "w2v1-table3-cbow",
        "reconstruction",
        "verified WMT substitute replaces unavailable LDC bytes",
    ),
    TargetDisposition(
        "w2v1-table3-skipgram",
        "reconstruction",
        "verified WMT substitute replaces unavailable LDC bytes",
    ),
    TargetDisposition(
        "w2v2-phrase-skipgram-1b-objectives",
        "reconstruction",
        "verified WMT substitute and deterministic phrase materialization",
    ),
    TargetDisposition(
        "w2v1-table3-nnlm",
        "unsupported",
        "the Word2Vec engine does not implement NNLM",
    ),
    TargetDisposition(
        "w2v1-google-news-nnlm-6b",
        "external_baseline",
        "published NNLM result is retained for comparison only",
    ),
    *(
        TargetDisposition(spec, "unsupported", reason)
        for spec, reason in (
            (
                "w2v1-google-news-cbow-scale",
                "original Google News bytes are unavailable",
            ),
            (
                "w2v1-google-news-skipgram-scale",
                "original Google News bytes are unavailable",
            ),
            ("w2v1-table6-cbow-6b", "DistBelief execution environment is unavailable"),
            (
                "w2v1-table6-skipgram-6b",
                "DistBelief execution environment is unavailable",
            ),
            (
                "w2v1-msr-sentence-skipgram",
                "evaluation files have no verified checksum",
            ),
            (
                "w2v2-word-skipgram-1b-objectives",
                "canonical word-only plan is not bound",
            ),
            ("w2v2-large-phrase-skipgram", "6B/33B source corpus is unavailable"),
        )
    ),
)


@dataclass(frozen=True)
class RunReportRecord:
    plan_revision: int
    planned_run_slot_id: str
    mlflow_run_id: str
    target_id: str
    metric: str
    mean: float
    ci_lower: float
    ci_upper: float
    coverage: float
    target_value: float | None
    tokens_per_second: float
    hardware: str

    def __post_init__(self) -> None:
        if self.plan_revision < 1:
            raise ValueError("plan_revision must be positive")
        if not self.planned_run_slot_id or not self.mlflow_run_id:
            raise ValueError("analysis requires explicit slot and MLflow run IDs")


def write_readiness_report(
    output_dir: Path, records: Iterable[RunReportRecord]
) -> tuple[Path, Path]:
    """Write stable CSV/Markdown deliverables under a suite analysis directory."""
    rows = sorted(records, key=lambda row: (row.target_id, row.planned_run_slot_id))
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "summary.csv"
    fields = tuple(RunReportRecord.__dataclass_fields__)
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
    md_path = output_dir / "summary.md"
    lines = [
        "# Word2Vec canonical results",
        "",
        "Throughput is reported with the hardware identity and is not compared as raw wall time across unlike hardware.",
        "",
        "| revision | target | slot | run | metric | mean (95% CI) | coverage | target | tokens/s | hardware |",
        "|---:|---|---|---|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        target = "—" if row.target_value is None else f"{row.target_value:.6g}"
        lines.append(
            f"| {row.plan_revision} | {row.target_id} | {row.planned_run_slot_id} | "
            f"{row.mlflow_run_id} | {row.metric} | "
            f"{row.mean:.6g} [{row.ci_lower:.6g}, {row.ci_upper:.6g}] | "
            f"{row.coverage:.2%} | {target} | {row.tokens_per_second:.6g} | "
            f"{row.hardware} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path, csv_path


__all__ = [
    "CLASSIFICATIONS",
    "DISPOSITIONS",
    "RunReportRecord",
    "TargetDisposition",
    "write_readiness_report",
]
