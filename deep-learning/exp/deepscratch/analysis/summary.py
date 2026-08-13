"""Scalar per-condition summaries over the canonical selected run set."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import csv
import json
from pathlib import Path

import numpy as np

from exp.framework.analysis.statistics import summarize_series

from ..identity import Variant, Volume
from .declarations import MetricDeclaration
from .input import AnalysisRun, StudyAnalysisInput
from .paths import result_stem


TRAINING_TIME = MetricDeclaration(
    "training_time_s",
    "seconds",
    "train",
    "run",
    ("runtime/train_total_s",),
    ("runtime/train_total_s",),
    protocols=("book-source-v1", "legacy"),
)

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


def summary_declarations(
    study_id: str,
    declared: Mapping[str, tuple[MetricDeclaration, ...]],
) -> tuple[MetricDeclaration, ...]:
    return (*declared.get(study_id, ()), TRAINING_TIME)


def write_study_summary(
    data: StudyAnalysisInput,
    *,
    volume: Volume,
    study_id: str,
    metrics: Iterable[MetricDeclaration],
    output_dir: Path,
    output_variants: tuple[Variant, ...],
    print_console: bool,
) -> Path:
    path = output_dir / f"{result_stem(volume, study_id, output_variants)}_summary.csv"
    rows: list[dict[str, object]] = []
    conditions = data.runs(
        tuple(condition.canonical_id for condition in data.declaration.conditions)
    )
    for condition_id, runs in conditions.items():
        for metric in metrics:
            values = _metric_values(data, runs, metric)
            rows.append(
                _summary_row(study_id, condition_id, data.variant, metric, runs, values)
            )
        parameter_values = [
            value
            for run in runs
            if (value := _parameter_count(data, run)) is not None
        ]
        rows.append(
            _summary_row(
                study_id,
                condition_id,
                data.variant,
                MetricDeclaration(
                    "parameter_count",
                    "parameters",
                    "model",
                    "run",
                    (),
                    (),
                ),
                runs,
                parameter_values,
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    if print_console:
        _print_rows(rows)
    return path


def print_summary_file(path: Path) -> None:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    _print_rows(rows)


def _metric_values(
    data: StudyAnalysisInput,
    runs: list[AnalysisRun],
    metric: MetricDeclaration,
) -> list[float]:
    values = []
    for run in runs:
        value = None
        for native_id in metric.native_ids(data.variant):
            value = data.metric_value(run, native_id)
            if value is not None:
                break
        if value is not None and np.isfinite(value):
            values.append(float(value) * metric.value_scale)
    return values


def _parameter_count(data: StudyAnalysisInput, run: AnalysisRun) -> int | None:
    path = data.artifact_file(run, "model/parameter_manifest.json")
    if path is None:
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            return None
        counts = [int(item["numel"]) for item in payload]
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    return sum(counts)


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
    return {
        **base,
        "seed_runs": stats.count,
        "mean": _format_number(stats.mean, decimals),
        "sample_standard_deviation": _format_number(
            stats.sample_standard_deviation, decimals
        ),
        "variance": _format_number(
            stats.sample_standard_deviation**2, decimals
        ),
        "minimum": _format_number(stats.minimum, decimals),
        "maximum": _format_number(stats.maximum, decimals),
        "availability": "available",
        "unavailable_reason": "",
    }


def _format_number(value: float, decimals: int) -> str:
    return f"{value:.{decimals}f}"


def _print_rows(rows: list[dict[str, object]]) -> None:
    current = None
    for row in rows:
        coordinate = (
            row["study_id"],
            row["canonical_condition_id"],
            row["variant"],
        )
        if coordinate != current:
            print(f"[{coordinate[0]}/{coordinate[1]}/{coordinate[2]}]")
            current = coordinate
        if row["availability"] != "available":
            print(f"{row['metric_id']}: unavailable")
            continue
        print(
            f"{row['metric_id']} ({row['unit']}): "
            f"{row['mean']} ± {row['sample_standard_deviation']} "
            f"(sample standard deviation; variance={row['variance']}; "
            f"n={row['seed_runs']})"
        )


__all__ = [
    "TRAINING_TIME",
    "print_summary_file",
    "summary_declarations",
    "write_study_summary",
]
