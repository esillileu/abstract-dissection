"""Text summaries of DS2 final metrics and training wall time."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import csv
from dataclasses import dataclass
from functools import partial
import json

import numpy as np

from exp.analyze import RunRef, artifact_file, artifact_rows

from .common import runs


@dataclass(frozen=True)
class MetricSummary:
    mean: float
    standard_deviation: float
    minimum: float
    maximum: float
    run_count: int


@dataclass(frozen=True)
class MetricSpec:
    name: str
    artifact_path: str
    row_metric: str
    source: str
    unit: str
    decimals: int
    scale: float = 1.0
    final_json_key: str | None = None


@dataclass(frozen=True)
class AnalysisSpec:
    group_id: str
    atomic_run_ids: tuple[str, ...]
    metric: MetricSpec


FINAL_LOSS = MetricSpec(
    "final_loss",
    "observations/source_curves.csv",
    "book_loss",
    "source_curve",
    "raw",
    3,
    final_json_key="final/train/book_loss",
)
FINAL_TRAIN_PERPLEXITY = MetricSpec(
    "final_train_perplexity",
    "observations/source_curves.csv",
    "perplexity",
    "source_curve",
    "raw",
    2,
    final_json_key="final/train/perplexity",
)
FINAL_TEST_PERPLEXITY = MetricSpec(
    "final_test_perplexity",
    "evaluations.csv",
    "perplexity",
    "terminal_test",
    "raw",
    2,
    final_json_key="final/test/perplexity",
)
FINAL_TEST_ACCURACY = MetricSpec(
    "final_test_accuracy",
    "observations/source_curves.csv",
    "exact_match_accuracy",
    "source_curve",
    "percent",
    2,
    100.0,
    "final/test/exact_match",
)


ANALYSES = {
    "e01": AnalysisSpec(
        "GT01",
        ("W2V-TOY-CBOW-FULL", "W2V-TOY-SKIPGRAM-FULL"),
        FINAL_LOSS,
    ),
    "e02": AnalysisSpec(
        "GT02",
        (
            "W2V-PTB-CBOW-NS",
            "W2V-PTB-SKIPGRAM-NS",
            "W2V-PTB-CBOW-FULL",
            "W2V-PTB-SKIPGRAM-FULL",
        ),
        FINAL_LOSS,
    ),
    "e03": AnalysisSpec("GT03", ("LM-SMALL-RNN",), FINAL_TRAIN_PERPLEXITY),
    "e04": AnalysisSpec("GT04", ("LM-LSTM",), FINAL_TEST_PERPLEXITY),
    "e05": AnalysisSpec(
        "GT05",
        ("LM-RNN-RECIPE", "LM-LSTM-RECIPE", "LM-BETTER-RECIPE"),
        FINAL_TEST_PERPLEXITY,
    ),
    "e06": AnalysisSpec(
        "GT06",
        ("SEQA-VAN-FWD", "SEQA-VAN-REV", "SEQA-PEEKY-FWD", "SEQA-PEEKY-REV"),
        FINAL_TEST_ACCURACY,
    ),
    "e07": AnalysisSpec(
        "GT07",
        ("SEQD-VAN-REV", "SEQD-PEEKY-REV", "SEQD-ATTN-REV"),
        FINAL_TEST_ACCURACY,
    ),
}


def _last_source_value(rows: list[dict[str, str]], metric: str) -> float | None:
    values = []
    for row in rows:
        if row.get("metric") != metric:
            continue
        try:
            values.append((float(row["plot_index"]), float(row["value"])))
        except (KeyError, TypeError, ValueError):
            continue
    return max(values)[1] if values else None


def _terminal_test_value(rows: list[dict[str, str]], metric: str) -> float | None:
    values = []
    for row in rows:
        if (
            row.get("axis") != "terminal"
            or row.get("split") != "test"
            or row.get("metric") != metric
        ):
            continue
        try:
            values.append((float(row["axis_step"]), float(row["value"])))
        except (KeyError, TypeError, ValueError):
            continue
    return max(values)[1] if values else None


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


def _values(
    client,
    run_refs: Sequence[RunRef],
    *,
    artifact_path: str,
    value_from_rows: Callable[[list[dict[str, str]]], float | None],
) -> np.ndarray:
    values = []
    for run in run_refs:
        value = value_from_rows(artifact_rows(client, run, artifact_path))
        if value is not None and np.isfinite(value):
            values.append(float(value))
    return np.asarray(values, dtype=float)


def _final_json_value(client, run: RunRef, key: str | None) -> float | None:
    if key is None:
        return None
    path = artifact_file(client, run, "metrics/final.json")
    if path is None:
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        value = float(payload[key])
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _performance_values(
    client,
    run_refs: Sequence[RunRef],
    metric: MetricSpec,
    value_from_rows: Callable[[list[dict[str, str]]], float | None],
) -> np.ndarray:
    values = []
    for run in run_refs:
        value = value_from_rows(
            artifact_rows(client, run, metric.artifact_path)
        )
        if value is None or not np.isfinite(value):
            value = _final_json_value(client, run, metric.final_json_key)
        if value is not None and np.isfinite(value):
            values.append(float(value))
    return np.asarray(values, dtype=float)


def _summarize(values: np.ndarray) -> MetricSummary | None:
    if not len(values):
        return None
    return MetricSummary(
        mean=float(values.mean()),
        standard_deviation=float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        minimum=float(values.min()),
        maximum=float(values.max()),
        run_count=len(values),
    )


def summarize_atomic_runs(
    client,
    run_refs: Sequence[RunRef],
    metric: MetricSpec,
) -> tuple[MetricSummary | None, MetricSummary | None]:
    if metric.source == "source_curve":
        value_from_rows = partial(_last_source_value, metric=metric.row_metric)
    elif metric.source == "terminal_test":
        value_from_rows = partial(_terminal_test_value, metric=metric.row_metric)
    else:
        raise ValueError(f"unknown metric source: {metric.source}")
    performance = _summarize(
        _performance_values(
            client,
            run_refs,
            metric,
            value_from_rows,
        )
    )
    training_time = _summarize(
        _values(
            client,
            run_refs,
            artifact_path="timing_windows.csv",
            value_from_rows=_training_seconds,
        )
    )
    return performance, training_time


def _format_summary(
    label: str,
    summary: MetricSummary | None,
    *,
    scale: float,
    decimals: int,
    unit_label: str,
) -> str:
    if summary is None:
        return f"{label}: no completed values"
    return (
        f"{label}{unit_label}: {summary.mean * scale:.{decimals}f} "
        f"± {summary.standard_deviation * scale:.{decimals}f}, "
        f"[{summary.minimum * scale:.{decimals}f}, "
        f"{summary.maximum * scale:.{decimals}f}], "
        f"n={summary.run_count}"
    )


def _display_values(
    summary: MetricSummary,
    *,
    scale: float,
    decimals: int,
) -> tuple[str, str, str, str]:
    return tuple(
        f"{value * scale:.{decimals}f}"
        for value in (
            summary.mean,
            summary.standard_deviation,
            summary.minimum,
            summary.maximum,
        )
    )


def _write_summaries(
    path,
    summaries: Mapping[
        str,
        tuple[MetricSpec, MetricSummary | None, MetricSummary | None],
    ],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "series",
        "metric",
        "seed_runs",
        "unit",
        "mean",
        "standard_deviation",
        "minimum",
        "maximum",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for atomic_run_id, (metric, performance, training_time) in summaries.items():
            for name, unit, decimals, summary in (
                (metric.name, metric.unit, metric.decimals, performance),
                ("training_time_s", "seconds", 1, training_time),
            ):
                values = (
                    None
                    if summary is None
                    else _display_values(
                        summary,
                        scale=metric.scale if name == metric.name else 1.0,
                        decimals=decimals,
                    )
                )
                writer.writerow(
                    {
                        "series": atomic_run_id,
                        "metric": name,
                        "seed_runs": "" if summary is None else summary.run_count,
                        "unit": unit,
                        "mean": "" if values is None else values[0],
                        "standard_deviation": "" if values is None else values[1],
                        "minimum": "" if values is None else values[2],
                        "maximum": "" if values is None else values[3],
                    }
                )


def render_summary(client, error_style, output, *, analysis_id: str):
    del error_style
    spec = ANALYSES[analysis_id]
    grouped_runs = runs(client, spec.group_id, list(spec.atomic_run_ids))
    summaries = {
        atomic_run_id: (
            spec.metric,
            *summarize_atomic_runs(client, grouped_runs[atomic_run_id], spec.metric),
        )
        for atomic_run_id in spec.atomic_run_ids
    }
    print(f"{analysis_id} summary (mean ± sample standard deviation; min-max)")
    for atomic_run_id, (metric, performance, training_time) in summaries.items():
        print(f"[{atomic_run_id}]")
        print(
            _format_summary(
                metric.name,
                performance,
                scale=metric.scale,
                decimals=metric.decimals,
                unit_label=" (%)" if metric.unit == "percent" else "",
            )
        )
        print(
            _format_summary(
                "training_time",
                training_time,
                scale=1.0,
                decimals=1,
                unit_label=" (s)",
            )
        )
    summary_path = output.with_suffix(".csv")
    _write_summaries(summary_path, summaries)
    return [summary_path]


FINAL_METRIC_RENDERERS = {
    analysis_id: partial(render_summary, analysis_id=analysis_id)
    for analysis_id in ANALYSES
}
