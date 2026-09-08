"""Base analysis orchestrator for loading multi-seed runs and generating summaries."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from ..stats.bootstrap import bootstrap_confidence_interval


@dataclass(frozen=True)
class NormalizedMetricSummary:
    metric_id: str
    condition_id: str
    mean_value: float
    std_value: float
    ci_lower: float
    ci_upper: float
    sample_size: int


class BaseAnalysisOrchestrator:
    """Base analysis orchestrator generating observations.csv and summary markdown."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_observations_csv(
        self,
        rows: Sequence[dict[str, Any]],
        filename: str = "observations.csv",
    ) -> Path:
        """Write tabular run metric observations to CSV."""
        if not rows:
            target = self.output_dir / filename
            target.write_text("", encoding="utf-8")
            return target

        target = self.output_dir / filename
        fieldnames = list(rows[0].keys())
        with target.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return target

    def summarize_metric_series(
        self, values: Sequence[float], metric_id: str, condition_id: str
    ) -> NormalizedMetricSummary:
        """Compute mean, standard deviation, and 95% bootstrap confidence interval."""
        if not values:
            return NormalizedMetricSummary(
                metric_id=metric_id,
                condition_id=condition_id,
                mean_value=0.0,
                std_value=0.0,
                ci_lower=0.0,
                ci_upper=0.0,
                sample_size=0,
            )

        mean_val = sum(values) / len(values)
        variance = sum((v - mean_val) ** 2 for v in values) / max(1, len(values) - 1)
        std_val = variance**0.5
        ci_low, ci_high = bootstrap_confidence_interval(values)

        return NormalizedMetricSummary(
            metric_id=metric_id,
            condition_id=condition_id,
            mean_value=mean_val,
            std_value=std_val,
            ci_lower=ci_low,
            ci_upper=ci_high,
            sample_size=len(values),
        )


__all__ = [
    "BaseAnalysisOrchestrator",
    "NormalizedMetricSummary",
]
