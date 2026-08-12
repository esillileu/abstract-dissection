"""Shared text summaries for final accuracy and training wall time."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import csv
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from exp.analyze import Curve, RunRef, aggregate, artifact_rows
from exp.ds1.final_gap import TRAIN_FULL_ACCURACY, TRAIN_TEST_ACCURACY_GAP

from .common import runs


ORIGINAL_DATA_ROOT = (
    Path(__file__).resolve().parents[2]
    / "ds1_original/results/legacy_cache/fixed_seed/data"
)
ORIGINAL_TRIAL_IDS = {
    "CNN-SIMPLE-BOOK": ("e06", "dlfs1.ch07.simple-convnet"),
    "CNN-DEEP-BOOK": ("e07", "dlfs1.ch08.deep-convnet"),
}
FINAL_ACCURACY_SOURCES = {
    "train_accuracy": ("train", "mnist-train-first-1000", "train"),
    "test_accuracy": ("test", "mnist-test-full", "test-full"),
}


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


def _logged_metric_summary(
    client,
    run_refs: Sequence[RunRef],
    metric_key: str,
) -> MetricSummary | None:
    values = []
    for run_ref in run_refs:
        try:
            value = float(client.get_run(run_ref.run_id).data.metrics[metric_key])
        except (AttributeError, KeyError, TypeError, ValueError):
            continue
        if np.isfinite(value):
            values.append(value)
    if not values:
        return None
    array = np.asarray(values, dtype=float)
    return MetricSummary(
        mean=float(array.mean()),
        standard_deviation=(
            float(array.std(ddof=1)) if len(array) > 1 else 0.0
        ),
        minimum=float(array.min()),
        maximum=float(array.max()),
        run_count=len(array),
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


def _final_sampled_accuracy(
    rows: list[dict[str, str]],
    *,
    split: str,
    evaluation_set_id: str,
) -> float | None:
    values = []
    for position, row in enumerate(rows):
        if (
            row.get("split") != split
            or row.get("evaluation_set_id") != evaluation_set_id
        ):
            continue
        try:
            step = float(row["axis_step"])
            accuracy = float(row["accuracy"])
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(step) and np.isfinite(accuracy):
            values.append((step, position, accuracy))
    return max(values)[2] if values else None


def _original_final_accuracy(
    atomic_run_id: str,
    split: str,
    *,
    original_data_root: Path = ORIGINAL_DATA_ROOT,
) -> float | None:
    try:
        experiment_id, trial_id = ORIGINAL_TRIAL_IDS[atomic_run_id]
    except KeyError:
        return None
    path = original_data_root / experiment_id / trial_id / "metrics.csv"
    try:
        with path.open(encoding="utf-8", newline="") as file:
            rows = list(csv.DictReader(file))
    except OSError:
        return None
    values = []
    for position, row in enumerate(rows):
        if row.get("split") != split:
            continue
        try:
            epoch = float(row["epoch"])
            accuracy = float(row["accuracy"])
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(epoch) and np.isfinite(accuracy):
            values.append((epoch, position, accuracy))
    return max(values)[2] if values else None


def _original_projected_training_seconds(
    atomic_run_id: str,
    *,
    original_data_root: Path = ORIGINAL_DATA_ROOT,
) -> float | None:
    try:
        experiment_id, _trial_id = ORIGINAL_TRIAL_IDS[atomic_run_id]
        payload = json.loads(
            (original_data_root.parent / "cupy_estimate.json").read_text(
                encoding="utf-8"
            )
        )
        result = next(
            item
            for item in payload["results"]
            if item.get("experiment_id") == experiment_id
        )
        value = float(result["projected_update_time_s"])
    except (
        OSError,
        json.JSONDecodeError,
        KeyError,
        StopIteration,
        TypeError,
        ValueError,
    ):
        return None
    return value if np.isfinite(value) else None


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


def accuracy_summaries_for_runs(
    client,
    grouped_runs: Mapping[str, Sequence[RunRef]],
) -> dict[str, MetricSummary | None]:
    """Summarize each seed's last plotted train and test accuracy."""

    summaries = {}
    for atomic_run_id, run_refs in grouped_runs.items():
        for metric_name, (
            split,
            evaluation_set_id,
            _original_split,
        ) in FINAL_ACCURACY_SOURCES.items():
            summaries[f"{atomic_run_id}/{metric_name}"] = _metric_summary(
                client,
                run_refs,
                lambda rows, split=split, evaluation_set_id=evaluation_set_id: (
                    _final_sampled_accuracy(
                        rows,
                        split=split,
                        evaluation_set_id=evaluation_set_id,
                    )
                ),
                artifact_path="evaluations.csv",
            )
        summaries[f"{atomic_run_id}/training_time_s"] = _metric_summary(
            client,
            run_refs,
            _training_seconds,
            artifact_path="timing_windows.csv",
        )
    return summaries


def full_accuracy_summaries_for_runs(
    client,
    grouped_runs: Mapping[str, Sequence[RunRef]],
) -> dict[str, MetricSummary | None]:
    """Summarize final full-train/full-test accuracy, gap, and train time."""

    summaries = {}
    for atomic_run_id, run_refs in grouped_runs.items():
        summaries[f"{atomic_run_id}/train_accuracy"] = _logged_metric_summary(
            client,
            run_refs,
            TRAIN_FULL_ACCURACY,
        )
        summaries[f"{atomic_run_id}/test_accuracy"] = _metric_summary(
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
        summaries[f"{atomic_run_id}/train_test_gap"] = _logged_metric_summary(
            client,
            run_refs,
            TRAIN_TEST_ACCURACY_GAP,
        )
    return summaries


def _format_compact_metric(
    name: str,
    summary: MetricSummary | None,
) -> str:
    if summary is None:
        return f"{name}: no completed runs"
    if name in {"train_accuracy", "test_accuracy", "train_test_gap"}:
        label, scale, decimals = f"{name} (%)", 100, 2
    elif name == "training_time":
        label, scale, decimals = "training_time (s)", 1, 1
    else:
        raise ValueError(f"unknown compact summary metric: {name}")
    return (
        f"{label}: {summary.mean * scale:.{decimals}f} ± "
        f"{summary.standard_deviation * scale:.{decimals}f} "
        f"(n={summary.run_count})"
    )


def _write_accuracy_comparisons(
    path: Path,
    atomic_run_ids: Sequence[str],
    summaries: Mapping[str, MetricSummary | None],
    *,
    original_data_root: Path,
    full_train_and_gap: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "series",
                "metric",
                "evaluation_set",
                "original",
                "original_kind",
                "seed_runs",
                "unit",
                "mean",
                "standard_deviation",
                "minimum",
                "maximum",
            ],
        )
        writer.writeheader()
        for atomic_run_id in atomic_run_ids:
            accuracy_sources = dict(FINAL_ACCURACY_SOURCES)
            if full_train_and_gap:
                accuracy_sources["train_accuracy"] = (
                    "train",
                    "mnist-train-full",
                    "train",
                )
            for metric_name, (
                _split,
                evaluation_set_id,
                original_split,
            ) in accuracy_sources.items():
                summary = summaries[f"{atomic_run_id}/{metric_name}"]
                original = _original_final_accuracy(
                    atomic_run_id,
                    original_split,
                    original_data_root=original_data_root,
                )
                display = None if summary is None else _display_values(
                    f"{atomic_run_id}/final_test_accuracy",
                    summary,
                )
                writer.writerow(
                    {
                        "series": atomic_run_id,
                        "metric": metric_name,
                        "evaluation_set": evaluation_set_id,
                        "original": (
                            "" if original is None else f"{original * 100:.2f}"
                        ),
                        "original_kind": "measured",
                        "seed_runs": "" if summary is None else summary.run_count,
                        "unit": "percent",
                        "mean": "" if display is None else display[1],
                        "standard_deviation": "" if display is None else display[2],
                        "minimum": "" if display is None else display[3],
                        "maximum": "" if display is None else display[4],
                    }
                )
            time_summary = summaries[f"{atomic_run_id}/training_time_s"]
            original_time = _original_projected_training_seconds(
                atomic_run_id,
                original_data_root=original_data_root,
            )
            time_display = None if time_summary is None else _display_values(
                f"{atomic_run_id}/training_time_s",
                time_summary,
            )
            writer.writerow(
                {
                    "series": atomic_run_id,
                    "metric": "training_time_s",
                    "evaluation_set": "",
                    "original": (
                        "" if original_time is None else f"{original_time:.1f}"
                    ),
                    "original_kind": "projected",
                    "seed_runs": (
                        "" if time_summary is None else time_summary.run_count
                    ),
                    "unit": "seconds",
                    "mean": "" if time_display is None else time_display[1],
                    "standard_deviation": (
                        "" if time_display is None else time_display[2]
                    ),
                    "minimum": "" if time_display is None else time_display[3],
                    "maximum": "" if time_display is None else time_display[4],
                }
            )
            if full_train_and_gap:
                gap_summary = summaries[f"{atomic_run_id}/train_test_gap"]
                gap_display = None if gap_summary is None else _display_values(
                    f"{atomic_run_id}/final_test_accuracy",
                    gap_summary,
                )
                writer.writerow(
                    {
                        "series": atomic_run_id,
                        "metric": "train_test_gap",
                        "evaluation_set": "mnist-train-full - mnist-test-full",
                        "original": "",
                        "original_kind": "",
                        "seed_runs": (
                            "" if gap_summary is None else gap_summary.run_count
                        ),
                        "unit": "percent",
                        "mean": "" if gap_display is None else gap_display[1],
                        "standard_deviation": (
                            "" if gap_display is None else gap_display[2]
                        ),
                        "minimum": "" if gap_display is None else gap_display[3],
                        "maximum": "" if gap_display is None else gap_display[4],
                    }
                )


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


def render_cross_group_summary(
    client,
    *,
    analysis_id: str,
    models: Sequence[tuple[str, str]],
    output,
):
    """Summarize atomic runs that intentionally come from different groups."""

    grouped_runs = {
        atomic_run_id: runs(client, group_id, [atomic_run_id])[atomic_run_id]
        for group_id, atomic_run_id in models
    }
    summaries = full_accuracy_summaries_for_runs(client, grouped_runs)
    print(f"{analysis_id} (mean ± sample standard deviation)")
    for _group_id, atomic_run_id in models:
        print(f"[{atomic_run_id}]")
        for metric_name in ("train_accuracy", "test_accuracy"):
            print(_format_compact_metric(
                metric_name,
                summaries[f"{atomic_run_id}/{metric_name}"],
            ))
        print(_format_compact_metric(
            "training_time",
            summaries[f"{atomic_run_id}/training_time_s"],
        ))
        print(_format_compact_metric(
            "train_test_gap",
            summaries[f"{atomic_run_id}/train_test_gap"],
        ))
    summary_path = output.with_suffix(".csv")
    _write_accuracy_comparisons(
        summary_path,
        [atomic_run_id for _group_id, atomic_run_id in models],
        summaries,
        original_data_root=ORIGINAL_DATA_ROOT,
        full_train_and_gap=True,
    )
    return [summary_path]


def render_accuracy_comparison_summary(
    client,
    *,
    analysis_id: str,
    group_id: str,
    atomic_run_ids: Sequence[str],
    output,
    original_data_root: Path = ORIGINAL_DATA_ROOT,
):
    grouped_runs = runs(client, group_id, list(atomic_run_ids))
    summaries = accuracy_summaries_for_runs(client, grouped_runs)
    print(f"{analysis_id} (mean ± sample standard deviation)")
    for atomic_run_id in atomic_run_ids:
        print(f"[{atomic_run_id}]")
        for metric_name in FINAL_ACCURACY_SOURCES:
            print(_format_compact_metric(
                metric_name,
                summaries[f"{atomic_run_id}/{metric_name}"],
            ))
        print(_format_compact_metric(
            "training_time",
            summaries[f"{atomic_run_id}/training_time_s"],
        ))
    summary_path = output.with_suffix(".csv")
    _write_accuracy_comparisons(
        summary_path,
        atomic_run_ids,
        summaries,
        original_data_root=original_data_root,
    )
    return [summary_path]
