"""Reference paper target metric definitions and comparison contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .orchestrator import NormalizedMetricSummary


@dataclass(frozen=True)
class PaperTargetMetric:
    """Published target metric from reference paper or baseline C binary execution."""

    metric_id: str
    paper_value: float
    description: str = ""
    paper_ref: str | None = None
    unit: str = ""
    tolerance: float = 0.0


@dataclass(frozen=True)
class TargetComparisonResult:
    """Structured evaluation comparing an observed metric series to a published paper target."""

    metric_id: str
    paper_value: float
    observed_mean: float
    observed_std: float
    ci_lower: float
    ci_upper: float
    sample_size: int
    absolute_error: float
    relative_error: float
    ci_contains_target: bool
    description: str = ""
    paper_ref: str | None = None


def compare_to_target(
    target: PaperTargetMetric, summary: NormalizedMetricSummary
) -> TargetComparisonResult:
    """Compare a normalized observation summary against a static paper target."""
    abs_err = abs(summary.mean_value - target.paper_value)
    rel_err = abs_err / abs(target.paper_value) if target.paper_value != 0.0 else 0.0
    ci_contains = summary.ci_lower <= target.paper_value <= summary.ci_upper

    return TargetComparisonResult(
        metric_id=target.metric_id,
        paper_value=target.paper_value,
        observed_mean=summary.mean_value,
        observed_std=summary.std_value,
        ci_lower=summary.ci_lower,
        ci_upper=summary.ci_upper,
        sample_size=summary.sample_size,
        absolute_error=abs_err,
        relative_error=rel_err,
        ci_contains_target=ci_contains,
        description=target.description,
        paper_ref=target.paper_ref,
    )


def compare_all_targets(
    targets: Sequence[PaperTargetMetric],
    summaries: Sequence[NormalizedMetricSummary],
) -> list[TargetComparisonResult]:
    """Batch compare matching targets and observation summaries by metric_id."""
    summary_map = {s.metric_id: s for s in summaries}
    results: list[TargetComparisonResult] = []
    for t in targets:
        if t.metric_id in summary_map:
            results.append(compare_to_target(t, summary_map[t.metric_id]))
    return results


__all__ = [
    "PaperTargetMetric",
    "TargetComparisonResult",
    "compare_all_targets",
    "compare_to_target",
]
