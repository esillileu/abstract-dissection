"""Row formatting and coordinate transformations for analysis summaries."""

from __future__ import annotations

from collections.abc import Mapping

from dlfs.analysis.declarations import MetricDeclaration
from dlfs.analysis.input import AnalysisRun
from dlfs.identity import Variant
from repro_core.analysis.statistics import summarize_series

FIELDS = (
    "study_id",
    "canonical_condition_id",
    "variant",
    "metric_id",
    "split",
    "unit",
    "seed_runs",
    "mean",
    "sample_standard_deviation",
    "variance",
    "minimum",
    "maximum",
    "availability",
    "unavailable_reason",
    "run_ids",
)


def _format_number(value: float, decimals: int) -> str:
    return f"{value:.{decimals}f}"


def _format_e14_number(value: float) -> str:
    if value != 0.0 and abs(value) < 0.01:
        return f"{value:.2e}"
    return f"{value:.2f}"


def _coordinate(row: Mapping[str, object]) -> tuple[object, object, object]:
    return (
        row["study_id"],
        row["canonical_condition_id"],
        row["variant"],
    )


def _formatted_summary(row: Mapping[str, object]) -> str:
    if row["availability"] != "available":
        reason = row["unavailable_reason"] or "reason unavailable"
        return f"{row['metric_id']} ({row['unit']}): unavailable — {reason}"
    return (
        f"{row['metric_id']} ({row['unit']}): "
        f"{row['mean']} ± {row['sample_standard_deviation']} "
        f"(sample standard deviation; variance={row['variance']}; "
        f"n={row['seed_runs']})"
    )


def _summary_row(
    study_id: str,
    condition_id: str,
    variant: Variant,
    metric: MetricDeclaration,
    runs: list[AnalysisRun],
    values: list[float | int],
) -> dict[str, object]:
    base = {
        "study_id": study_id,
        "canonical_condition_id": condition_id,
        "variant": variant.value,
        "metric_id": metric.metric_id,
        "split": metric.split,
        "unit": metric.unit,
        "run_ids": ",".join(run.run_id for run in runs),
    }
    if not values:
        return {
            **base,
            "seed_runs": 0,
            "mean": "",
            "sample_standard_deviation": "",
            "variance": "",
            "minimum": "",
            "maximum": "",
            "availability": "unavailable",
            "unavailable_reason": "metric or artifact is absent from selected runs",
        }
    stats = summarize_series(values)
    decimals = 0 if metric.metric_id == "parameter_count" else 2
    if study_id == "e08" and metric.metric_id in {
        "train_accuracy",
        "test_accuracy",
    }:
        decimals = 3
    formatter = (
        _format_e14_number
        if study_id == "e14" and metric.split == "gradient_check"
        else lambda value: _format_number(value, decimals)
    )
    return {
        **base,
        "seed_runs": stats.count,
        "mean": formatter(stats.mean),
        "sample_standard_deviation": formatter(stats.sample_standard_deviation),
        "variance": formatter(stats.sample_standard_deviation**2),
        "minimum": formatter(stats.minimum),
        "maximum": formatter(stats.maximum),
        "availability": "available",
        "unavailable_reason": "",
    }
