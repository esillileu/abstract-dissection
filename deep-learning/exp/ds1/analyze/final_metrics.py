"""Shared text summary for final full-test accuracy and training wall time."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import csv
from dataclasses import dataclass

import numpy as np

from exp.analyze import Curve, RunRef, aggregate, artifact_rows

from .common import runs


@dataclass(frozen=True)
class MetricSummary:
    mean: float
    standard_deviation: float
    minimum: float
    maximum: float
    run_count: int


def _scalar_values(
    client,
    run_refs: Sequence[RunRef],
    value_from_run: Callable[[list[dict[str, str]]], float | None],
    *,
    artifact_path: str,
) -> np.ndarray:
    values = []
    for run in run_refs:
        value = value_from_run(artifact_rows(client, run, artifact_path))
        if value is not None and np.isfinite(value):
            values.append(float(value))
    return np.asarray(values, dtype=float)


def _scalar_curve(
    client,
    run_refs: Sequence[RunRef],
    value_from_run: Callable[[list[dict[str, str]]], float | None],
    *,
    artifact_path: str,
) -> Curve:
    values = _scalar_values(
        client,
        run_refs,
        value_from_run,
        artifact_path=artifact_path,
    )
    return aggregate([{0.0: value} for value in values])


def final_test_accuracy_curve(client, run_refs: Sequence[RunRef]) -> Curve:
    """Aggregate each seed's last full-test accuracy."""

    return _scalar_curve(
        client,
        run_refs,
        _final_accuracy,
        artifact_path="evaluations.csv",
    )


def training_time_curve(client, run_refs: Sequence[RunRef]) -> Curve:
    """Aggregate per-seed training wall time, excluding evaluation windows."""

    return _scalar_curve(
        client,
        run_refs,
        _training_seconds,
        artifact_path="timing_windows.csv",
    )


def curves_for_runs(
    client,
    grouped_runs: Mapping[str, Sequence[RunRef]],
) -> dict[str, Curve]:
    curves = {}
    for atomic_run_id, run_refs in grouped_runs.items():
        curves[f"{atomic_run_id}/final_test_accuracy"] = final_test_accuracy_curve(
            client,
            run_refs,
        )
        curves[f"{atomic_run_id}/training_time_s"] = training_time_curve(
            client,
            run_refs,
        )
    return curves


def _metric_summary(
    client,
    run_refs: Sequence[RunRef],
    value_from_run: Callable[[list[dict[str, str]]], float | None],
    *,
    artifact_path: str,
) -> MetricSummary | None:
    values = _scalar_values(
        client,
        run_refs,
        value_from_run,
        artifact_path=artifact_path,
    )
    if not len(values):
        return None
    return MetricSummary(
        mean=float(values.mean()),
        standard_deviation=float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        minimum=float(values.min()),
        maximum=float(values.max()),
        run_count=len(values),
    )


def _final_accuracy(rows: list[dict[str, str]]) -> float | None:
    values = []
    for row in rows:
        if (
            row.get("split") == "test"
            and row.get("evaluation_set_id") == "mnist-test-full"
        ):
            try:
                values.append(float(row["accuracy"]))
            except (KeyError, TypeError, ValueError):
                continue
    return values[-1] if values else None


def _training_seconds(rows: list[dict[str, str]]) -> float | None:
    values = []
    for row in rows:
        try:
            value = float(row["train_wall_time_ns"])
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(value):
            values.append(value)
    return sum(values) / 1_000_000_000 if values else None


def summaries_for_runs(
    client,
    grouped_runs: Mapping[str, Sequence[RunRef]],
) -> dict[str, MetricSummary | None]:
    summaries = {}
    for atomic_run_id, run_refs in grouped_runs.items():
        summaries[f"{atomic_run_id}/final_test_accuracy"] = _metric_summary(
            client,
            run_refs,
            _final_accuracy,
            artifact_path="evaluations.csv",
        )
        summaries[f"{atomic_run_id}/training_time_s"] = _metric_summary(
            client,
            run_refs,
            _training_seconds,
            artifact_path="timing_windows.csv",
        )
    return summaries


def _format_metric(name: str, summary: MetricSummary | None) -> str:
    if summary is None:
        return f"{name}: no completed runs"
    if name == "final_test_accuracy":
        scale = 100
        decimals = 2
        label = "final_test_accuracy (%)"
    elif name == "training_time_s":
        scale = 1
        decimals = 1
        label = "training_time (s)"
    else:
        raise ValueError(f"unknown final metric: {name}")
    return (
        f"{label}: {summary.mean * scale:.{decimals}f} "
        f"± {summary.standard_deviation * scale:.{decimals}f}, "
        f"[{summary.minimum * scale:.{decimals}f}, "
        f"{summary.maximum * scale:.{decimals}f}], "
        f"n={summary.run_count}"
    )


def _display_values(name: str, summary: MetricSummary) -> tuple[str, str, str, str, str]:
    if name.endswith("/final_test_accuracy"):
        scale, decimals, unit = 100, 2, "percent"
    elif name.endswith("/training_time_s"):
        scale, decimals, unit = 1, 1, "seconds"
    else:
        raise ValueError(f"unknown final metric series: {name}")
    values = (
        summary.mean,
        summary.standard_deviation,
        summary.minimum,
        summary.maximum,
    )
    formatted = tuple(f"{value * scale:.{decimals}f}" for value in values)
    return unit, *formatted


def _write_summaries(path, summaries: Mapping[str, MetricSummary | None]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "series",
                "seed_runs",
                "unit",
                "mean",
                "standard_deviation",
                "minimum",
                "maximum",
            ],
        )
        writer.writeheader()
        for name, summary in summaries.items():
            display = None if summary is None else _display_values(name, summary)
            writer.writerow(
                {
                    "series": name,
                    "seed_runs": "" if summary is None else summary.run_count,
                    "unit": "" if display is None else display[0],
                    "mean": "" if display is None else display[1],
                    "standard_deviation": "" if display is None else display[2],
                    "minimum": "" if display is None else display[3],
                    "maximum": "" if display is None else display[4],
                }
            )


def render_summary(
    client,
    *,
    analysis_id: str,
    group_id: str,
    atomic_run_ids: Sequence[str],
    output,
):
    grouped_runs = runs(client, group_id, list(atomic_run_ids))
    summaries = summaries_for_runs(client, grouped_runs)
    print(f"{analysis_id} (mean ± sample standard deviation; min-max)")
    for atomic_run_id in atomic_run_ids:
        print(f"[{atomic_run_id}]")
        print(
            _format_metric(
                "final_test_accuracy",
                summaries[f"{atomic_run_id}/final_test_accuracy"],
            )
        )
        print(
            _format_metric(
                "training_time_s",
                summaries[f"{atomic_run_id}/training_time_s"],
            )
        )
    summary_path = output.with_suffix(".csv")
    _write_summaries(summary_path, summaries)
    return [summary_path]
