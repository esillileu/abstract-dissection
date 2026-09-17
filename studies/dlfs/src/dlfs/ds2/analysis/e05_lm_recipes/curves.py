"""Curve loading, filtering, and metric aggregation for e05 LM recipes."""

from __future__ import annotations

import numpy as np

from dlfs.analysis.input import histories_from_artifact
from repro_core.analysis.core import aggregate

from ..common import runs


def _evaluation_curve(client, run_refs, *, split, axis):
    histories = histories_from_artifact(
        client,
        run_refs,
        artifact_path="evaluations.csv",
        x="axis_step",
        y="value",
        row_filter=lambda row: (
            row.get("axis") == axis
            and row.get("split") == split
            and row.get("metric") == "perplexity"
        ),
        x_value=(lambda _row: 0.0) if axis == "terminal" else None,
    )
    if not histories and axis == "epoch":
        histories = histories_from_artifact(
            client,
            run_refs,
            artifact_path="raw/metrics.csv",
            x="epoch",
            y="perplexity",
            row_filter=lambda row: row.get("split") == split,
        )
    return aggregate(histories)


def _finite_extrema(curves):
    values = [
        values
        for curve in curves
        for values in (curve.minimum, curve.maximum)
        if len(values)
    ]
    if not values:
        return None
    combined = np.concatenate(values)
    finite = combined[np.isfinite(combined)]
    if not len(finite):
        return None
    return float(finite.min()), float(finite.max())


def _padded_limits(extrema, *, padding=0.06):
    if extrema is None:
        return None
    minimum, maximum = extrema
    span = maximum - minimum
    margin = max(span * padding, abs(maximum) * 0.01, 1.0)
    return minimum - margin, maximum + margin


def _evaluation_curves(client, atomic_ids, *, split, axis):
    grouped = runs(client, "GT05", list(atomic_ids))
    return {
        f"{atomic}/valid": _evaluation_curve(
            client, grouped[atomic], split=split, axis=axis
        )
        for atomic in atomic_ids
    }


def _valid_curves(client, atomic_ids):
    curves = _evaluation_curves(client, atomic_ids, split="valid", axis="epoch")
    return {key: curve for key, curve in curves.items()}


def _train_curves(client, atomic_ids):
    grouped = runs(client, "GT05", list(atomic_ids))
    return {
        f"{atomic}/train": aggregate(
            [
                {epoch: sum(values) / len(values) for epoch, values in by_epoch.items()}
                for run in grouped[atomic]
                for by_epoch in [_train_ppl_by_epoch(client, run)]
                if by_epoch
            ]
        )
        for atomic in atomic_ids
    }


def _train_ppl_by_epoch(client, run):
    by_epoch = {}
    rows = client.artifact_rows(run, "observations/source_curves.csv")
    if rows:
        rows = [row for row in rows if row.get("metric") == "perplexity"]
        epoch_key = "epoch_end"
    else:
        rows = [
            row
            for row in client.artifact_rows(run, "raw/metrics.csv")
            if row.get("split") == "train" and row.get("perplexity")
        ]
        epoch_key = "epoch"
    for row in rows:
        try:
            epoch = float(row[epoch_key])
            value = float(row["value"] if "value" in row else row["perplexity"])
        except (KeyError, TypeError, ValueError):
            continue
        by_epoch.setdefault(epoch, []).append(value)
    return by_epoch
