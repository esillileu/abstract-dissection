"""Figure-free summaries for DS1 training and observation experiments."""

from __future__ import annotations

from collections.abc import Sequence
import csv
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np

from exp.analyze import RunRef, artifact_rows

from .common import runs
from .e09_optimizer_trajectory import DEFINITIONS as TRAJECTORY_DEFINITIONS
from .e10_activation import ATOMIC_IDS as ACTIVATION_ATOMIC_IDS
from .final_metrics import MetricSummary, _training_seconds


@dataclass(frozen=True)
class MetricSpec:
    name: str
    artifact_path: str
    step: str
    value: str
    unit: str
    decimals: int
    scale: float = 1.0
    filters: tuple[tuple[str, str], ...] = ()
    series_suffix: str = ""


@dataclass(frozen=True)
class AnalysisSpec:
    group_id: str
    atomic_run_ids: tuple[str, ...]
    metrics: tuple[MetricSpec, ...]
    include_training_time: bool = True


FINAL_LOSS = MetricSpec(
    "final_train_objective",
    "updates.csv",
    "update",
    "loss",
    "raw",
    3,
)
FINAL_TRAIN_ACCURACY = MetricSpec(
    "final_train_accuracy",
    "evaluations.csv",
    "axis_step",
    "accuracy",
    "percent",
    2,
    scale=100.0,
    filters=(("split", "train"),),
)
FINAL_TEST_ACCURACY = MetricSpec(
    "final_test_accuracy",
    "evaluations.csv",
    "axis_step",
    "accuracy",
    "percent",
    2,
    scale=100.0,
    filters=(("split", "test"),),
)


def _activation_metrics() -> tuple[MetricSpec, ...]:
    metrics = []
    for layer in range(1, 6):
        for name, unit, decimals in (
            ("mean", "raw", 4),
            ("std", "raw", 4),
            ("min", "raw", 4),
            ("max", "raw", 4),
            ("zero_ratio", "ratio", 4),
        ):
            metrics.append(
                MetricSpec(
                    f"activation_{name}",
                    "observations/activation_summary.csv",
                    "layer",
                    name,
                    unit,
                    decimals,
                    filters=(("layer", str(layer)),),
                    series_suffix=f"/layer-{layer}",
                )
            )
    return tuple(metrics)


ANALYSES = {
    "e01": AnalysisSpec(
        "GT01",
        ("MLP-OPT-SGD", "MLP-OPT-MOMENTUM", "MLP-OPT-ADAGRAD", "MLP-OPT-ADAM"),
        (FINAL_LOSS,),
    ),
    "e02": AnalysisSpec(
        "GT02",
        ("MLP-INIT-STD001", "MLP-INIT-XAVIER", "MLP-INIT-HE"),
        (FINAL_LOSS,),
    ),
    "e03": AnalysisSpec(
        "GT03",
        ("REG-WD-OFF", "REG-WD-01"),
        (FINAL_TRAIN_ACCURACY, FINAL_TEST_ACCURACY),
    ),
    "e04": AnalysisSpec(
        "GT04",
        ("REG-DROPOUT-OFF", "REG-DROPOUT-ON-02"),
        (FINAL_TRAIN_ACCURACY, FINAL_TEST_ACCURACY),
    ),
    "e05": AnalysisSpec(
        "GT05",
        tuple(
            f"BN-SCALE-{index:02d}-{state}"
            for index in range(1, 17)
            for state in ("ON", "OFF")
        ),
        (FINAL_TRAIN_ACCURACY,),
    ),
    "e09": AnalysisSpec(
        "GO01",
        tuple(item[0] for item in TRAJECTORY_DEFINITIONS),
        tuple(
            MetricSpec(
                f"final_{name}", "observations/trajectory.csv", "update", name, "raw", 4
            )
            for name in ("x", "y", "objective")
        ),
        include_training_time=False,
    ),
    "e10": AnalysisSpec(
        "GO02",
        tuple(ACTIVATION_ATOMIC_IDS),
        _activation_metrics(),
        include_training_time=False,
    ),
}


def _last_value(rows: list[dict[str, str]], metric: MetricSpec) -> float | None:
    values = []
    for position, row in enumerate(rows):
        if any(row.get(key) != value for key, value in metric.filters):
            continue
        try:
            step = float(row[metric.step])
            value = float(row[metric.value])
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(step) and np.isfinite(value):
            values.append((step, position, value))
    return max(values)[2] if values else None


def _summarize(values: Sequence[float]) -> MetricSummary | None:
    array = np.asarray(values, dtype=float)
    if not len(array):
        return None
    return MetricSummary(
        mean=float(array.mean()),
        standard_deviation=float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        minimum=float(array.min()),
        maximum=float(array.max()),
        run_count=len(array),
    )


def summarize_metric(
    client,
    run_refs: Sequence[RunRef],
    metric: MetricSpec,
) -> MetricSummary | None:
    values = []
    for run in run_refs:
        value = _last_value(
            artifact_rows(client, run, metric.artifact_path),
            metric,
        )
        if value is not None:
            values.append(value)
    return _summarize(values)


def _training_time_summary(
    client,
    run_refs: Sequence[RunRef],
) -> MetricSummary | None:
    values = []
    for run in run_refs:
        value = _training_seconds(artifact_rows(client, run, "timing_windows.csv"))
        if value is not None:
            values.append(value)
    return _summarize(values)


def _formatted_values(
    summary: MetricSummary | None,
    *,
    scale: float,
    decimals: int,
) -> tuple[str, str, str, str]:
    if summary is None:
        return "", "", "", ""
    return tuple(
        f"{value * scale:.{decimals}f}"
        for value in (
            summary.mean,
            summary.standard_deviation,
            summary.minimum,
            summary.maximum,
        )
    )


def _print_summary(
    name: str, metric: MetricSpec, summary: MetricSummary | None
) -> None:
    if summary is None:
        print(f"{name}: no completed values")
        return
    mean, deviation, minimum, maximum = _formatted_values(
        summary,
        scale=metric.scale,
        decimals=metric.decimals,
    )
    unit = " (%)" if metric.unit == "percent" else ""
    print(
        f"{name}{unit}: {mean} ± {deviation}, "
        f"[{minimum}, {maximum}], n={summary.run_count}"
    )


def _write_rows(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "series",
        "metric",
        "seed_runs",
        "unit",
        "mean",
        "standard_deviation",
        "minimum",
        "maximum",
    )
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def render_summary(client, error_style, output: Path, *, analysis_id: str):
    del error_style
    spec = ANALYSES[analysis_id]
    grouped = runs(client, spec.group_id, list(spec.atomic_run_ids))
    output_rows: list[dict[str, object]] = []
    print(f"{analysis_id} summary (mean ± sample standard deviation; min-max)")
    for atomic_run_id in spec.atomic_run_ids:
        print(f"[{atomic_run_id}]")
        for metric in spec.metrics:
            summary = summarize_metric(client, grouped[atomic_run_id], metric)
            series = f"{atomic_run_id}{metric.series_suffix}"
            display_name = (
                f"{metric.series_suffix.removeprefix('/')}/{metric.name}"
                if metric.series_suffix
                else metric.name
            )
            _print_summary(display_name, metric, summary)
            mean, deviation, minimum, maximum = _formatted_values(
                summary,
                scale=metric.scale,
                decimals=metric.decimals,
            )
            output_rows.append(
                {
                    "series": series,
                    "metric": metric.name,
                    "seed_runs": "" if summary is None else summary.run_count,
                    "unit": metric.unit,
                    "mean": mean,
                    "standard_deviation": deviation,
                    "minimum": minimum,
                    "maximum": maximum,
                }
            )
        if spec.include_training_time:
            summary = _training_time_summary(client, grouped[atomic_run_id])
            timing_metric = MetricSpec("training_time_s", "", "", "", "seconds", 1)
            _print_summary("training_time", timing_metric, summary)
            mean, deviation, minimum, maximum = _formatted_values(
                summary,
                scale=1.0,
                decimals=1,
            )
            output_rows.append(
                {
                    "series": atomic_run_id,
                    "metric": "training_time_s",
                    "seed_runs": "" if summary is None else summary.run_count,
                    "unit": "seconds",
                    "mean": mean,
                    "standard_deviation": deviation,
                    "minimum": minimum,
                    "maximum": maximum,
                }
            )
    return [_write_rows(output.with_suffix(".csv"), output_rows)]


SUMMARY_RENDERERS = {
    analysis_id: partial(render_summary, analysis_id=analysis_id)
    for analysis_id in ANALYSES
}
