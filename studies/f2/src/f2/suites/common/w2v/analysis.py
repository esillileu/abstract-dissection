"""Stable W2V run summaries, paper-target comparisons, and suite-level table helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence, TypeVar

from f2.common.analysis.orchestrator import BaseAnalysisOrchestrator
from f2.common.analysis.tables import (
    BaseEvaluationCache,
    format_duration,
    format_duration_mean_sd,
    format_mean_sd,
    observed_training_seconds,
    write_run_records_csv,
    write_table_summary,
)
from f2.common.analysis.targets import (
    PaperTargetMetric,
    TargetComparisonResult,
    compare_to_target,
)
from f2.suites.common.w2v.evaluation import EvaluationResult

T = TypeVar("T")


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
        [run.require_score() for run in runs], metric_id, condition_id
    )
    comparison = compare_to_target(target, normalized) if target is not None else None
    return EvaluationSummary(
        metric_id=metric_id,
        mean_value=normalized.mean_value,
        std_value=normalized.std_value,
        ci_lower=normalized.ci_lower,
        ci_upper=normalized.ci_upper,
        sample_size=normalized.sample_size,
        mean_coverage=sum(run.coverage for run in runs if run.coverage is not None)
        / len(runs),
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


class W2VTableEvaluationCache(BaseEvaluationCache[T]):
    """W2V suite-level evaluation cache for questions-based benchmarks."""

    def __init__(
        self,
        root: Path,
        protocol: str,
        result_cls: type[T],
        *,
        questions_sha256: str,
        extra_signature: dict[str, Any] | None = None,
    ) -> None:
        sig = {
            "schema_version": 2,
            "evaluation_protocol": protocol,
            "questions_sha256": questions_sha256,
        }
        if extra_signature:
            sig.update(extra_signature)
        super().__init__(root, signature=sig, result_cls=result_cls)


def format_accuracy_and_time(
    records: list[Any],
    acc_field: str = "total_accuracy_percent",
    time_field: str = "observed_training_seconds",
    precision: int = 1,
) -> tuple[str, str]:
    """Calculate mean ± SD for accuracy and training time."""
    acc_vals = [float(getattr(r, acc_field)) for r in records]
    time_vals = [float(getattr(r, time_field)) for r in records]
    acc_str = format_mean_sd(acc_vals, precision=precision)
    time_str = format_duration_mean_sd(time_vals)
    return acc_str, time_str


__all__ = [
    "BaseEvaluationCache",
    "EvaluationSummary",
    "W2VTableEvaluationCache",
    "format_accuracy_and_time",
    "format_duration",
    "format_duration_mean_sd",
    "format_mean_sd",
    "observed_training_seconds",
    "summarize_evaluations",
    "summary_record",
    "write_run_records_csv",
    "write_table_summary",
]
