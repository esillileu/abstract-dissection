"""Metric extraction and historical fallback recovery for study analyses."""

from __future__ import annotations

import json

import numpy as np

from dlfs.analysis.declarations import MetricDeclaration
from dlfs.analysis.input import AnalysisRun, StudyAnalysisInput
from dlfs.identity import Variant


def _metric_values(
    data: StudyAnalysisInput,
    study_id: str,
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
        if value is None:
            value = _curve_summary_fallback(data, study_id, run, metric)
        if value is None:
            value = _original_ds1_accuracy_fallback(data, study_id, run, metric)
        if value is not None and np.isfinite(value):
            values.append(float(value) * metric.value_scale)
    return values


def _curve_summary_fallback(
    data: StudyAnalysisInput,
    study_id: str,
    run: AnalysisRun,
    metric: MetricDeclaration,
) -> float | None:
    """Use the last recorded evaluation for studies without a final scalar."""
    if study_id not in {"e08", "e15"} or metric.metric_id != "train_accuracy":
        return None
    try:
        histories = data.metric_histories([run], "update/eval_train/accuracy")
    except (AttributeError, KeyError, OSError, TypeError, ValueError):
        return None
    if not histories or not histories[0]:
        return None
    history = histories[0]
    return float(history[max(history)])


def _original_ds1_accuracy_fallback(
    data: StudyAnalysisInput,
    study_id: str,
    run: AnalysisRun,
    metric: MetricDeclaration,
) -> float | None:
    """Recover missing original accuracy from the raw evaluation CSV."""
    if (
        study_id not in {"e03", "e04", "e06", "e07"}
        or run.variant is not Variant.ORIGINAL
        or metric.metric_id not in {"train_accuracy", "test_accuracy"}
    ):
        return None
    rows = data.artifact_rows(run, "raw/metrics.csv")
    candidate_splits = (metric.split,)
    if metric.metric_id == "test_accuracy":
        candidate_splits = ("test-full", "test")
    for split in candidate_splits:
        values = []
        for row in rows:
            if row.get("split") != split:
                continue
            try:
                value = float(row["accuracy"])
            except (KeyError, TypeError, ValueError):
                continue
            if np.isfinite(value):
                values.append(value)
        if values:
            return values[-1]
    return None


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
