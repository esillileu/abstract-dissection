"""Stable W2V run summaries and paper-target comparisons."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

from f2.common.analysis.orchestrator import BaseAnalysisOrchestrator
from f2.common.analysis.targets import (
    PaperTargetMetric,
    TargetComparisonResult,
    compare_to_target,
)

from .evaluation import EvaluationResult


@dataclass(frozen=True)
class EvaluationSummary:
    metric_id: str
    mean_value: float
    std_value: float
    ci_lower: float
    ci_upper: float
    sample_size: int
    mean_coverage: float
    paper_comparison: TargetComparisonResult | None


def summarize_evaluations(
    runs: Sequence[EvaluationResult],
    *,
    condition_id: str,
    target: PaperTargetMetric | None = None,
) -> EvaluationSummary:
    """Build a deterministic multi-run summary including coverage and optional target error."""
    if not runs:
        raise ValueError("at least one evaluation result is required")
    metric_id = runs[0].metric_id
    if any(run.metric_id != metric_id for run in runs):
        raise ValueError("evaluation results must share one metric_id")
    if target is not None and target.metric_id != metric_id:
        raise ValueError("paper target metric_id does not match evaluation results")
    normalized = BaseAnalysisOrchestrator.summarize_metric_series(
        [run.score for run in runs], metric_id, condition_id
    )
    comparison = compare_to_target(target, normalized) if target is not None else None
    return EvaluationSummary(
        metric_id=metric_id,
        mean_value=normalized.mean_value,
        std_value=normalized.std_value,
        ci_lower=normalized.ci_lower,
        ci_upper=normalized.ci_upper,
        sample_size=normalized.sample_size,
        mean_coverage=sum(run.coverage for run in runs) / len(runs),
        paper_comparison=comparison,
    )


def summary_record(summary: EvaluationSummary) -> dict[str, object]:
    """Return a JSON/CSV-ready record with a fixed field order."""
    record = asdict(summary)
    comparison = record.pop("paper_comparison")
    record["paper_value"] = None if comparison is None else comparison["paper_value"]
    record["absolute_error"] = (
        None if comparison is None else comparison["absolute_error"]
    )
    record["relative_error"] = (
        None if comparison is None else comparison["relative_error"]
    )
    record["ci_contains_target"] = (
        None if comparison is None else comparison["ci_contains_target"]
    )
    return record


__all__ = ["EvaluationSummary", "summarize_evaluations", "summary_record"]
