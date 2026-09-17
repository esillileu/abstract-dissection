from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from repro_core.plotting.theme import ACCENT_COLORS, MUTED

_PLOT_STYLES = {
    "implemented-cbow-fs": ("Full Softmax", ACCENT_COLORS[0], "o", "-"),
    "implemented-cbow-ns": (
        "Negative Sampling",
        ACCENT_COLORS[1],
        "s",
        "--",
    ),
    "implemented-cbow-fused-ns": (
        "Fused Negative Sampling",
        ACCENT_COLORS[3],
        "^",
        "-.",
    ),
    "implemented-skipgram-fs": (
        "Full Softmax",
        ACCENT_COLORS[0],
        "o",
        "-",
    ),
    "implemented-skipgram-ns": (
        "Negative Sampling",
        ACCENT_COLORS[1],
        "s",
        "--",
    ),
    "implemented-skipgram-fused-ns": (
        "Fused Negative Sampling",
        ACCENT_COLORS[3],
        "^",
        "-.",
    ),
}


def render_scaling(payload: dict[str, object]):
    """Render one repository-themed vocabulary/runtime figure per device."""
    rows = payload.get("results")
    metadata = payload.get("metadata")
    if not isinstance(rows, list) or not isinstance(metadata, dict):
        raise ValueError("invalid vocabulary-size scaling payload")
    selected = [
        row
        for row in rows
        if isinstance(row, dict)
        and row.get("status") == "ok"
        and row.get("condition") in _PLOT_STYLES
    ]
    models = [
        model
        for model in ("CBOW", "SkipGram")
        if any(row.get("model") == model for row in selected)
    ]
    if not models:
        raise ValueError("vocabulary-size scaling has no plottable results")

    figure, axes = plt.subplots(
        1,
        len(models),
        figsize=(6.4 if len(models) == 1 else 10.0, 4.8),
        squeeze=False,
    )
    device = str(metadata.get("device", "unknown device"))
    device_label = (
        "GPU"
        if device.startswith("cuda:")
        else "CPU"
        if device.startswith("cpu")
        else device
    )
    for axis, model in zip(axes[0], models, strict=True):
        _plot_model(
            axis,
            selected,
            model,
            device_label,
            timing_source=str(metadata.get("timing_source", "window")),
            title=True,
        )
    return figure


def render_individual_scaling(
    payload: dict[str, object],
) -> list[tuple[str, plt.Figure]]:
    """Render one untitled figure for each model represented in the payload."""
    rows = payload.get("results")
    metadata = payload.get("metadata")
    if not isinstance(rows, list) or not isinstance(metadata, dict):
        raise ValueError("invalid vocabulary-size scaling payload")
    selected = [
        row
        for row in rows
        if isinstance(row, dict)
        and row.get("status") == "ok"
        and row.get("condition") in _PLOT_STYLES
    ]
    models = [
        model
        for model in ("CBOW", "SkipGram")
        if any(row.get("model") == model for row in selected)
    ]
    if not models:
        raise ValueError("vocabulary-size scaling has no plottable results")
    device = str(metadata.get("device", "unknown device"))
    device_label = (
        "GPU"
        if device.startswith("cuda:")
        else "CPU"
        if device.startswith("cpu")
        else device
    )
    figures = []
    for model in models:
        figure, axis = plt.subplots(figsize=(6.4, 4.8))
        _plot_model(
            axis,
            selected,
            model,
            device_label,
            timing_source=str(metadata.get("timing_source", "window")),
            title=False,
        )
        figures.append((model, figure))
    return figures


def _plot_model(
    axis,
    selected,
    model: str,
    device_label: str,
    *,
    timing_source: str,
    title: bool,
) -> None:
    if timing_source not in {"window", "event"}:
        raise ValueError("timing source must be 'window' or 'event'")
    model_rows = [row for row in selected if row.get("model") == model]
    conditions = [
        condition
        for condition in _PLOT_STYLES
        if any(row.get("condition") == condition for row in model_rows)
    ]
    for condition in conditions:
        label, color, marker, linestyle = _PLOT_STYLES[condition]
        condition_rows = sorted(
            (row for row in model_rows if row.get("condition") == condition),
            key=lambda row: int(row["vocab_size"]),
        )
        vocabulary = np.asarray(
            [int(row["vocab_size"]) for row in condition_rows], dtype=float
        )
        update_ms = np.asarray(
            [
                float(row["update_ms"])
                if timing_source == "window"
                else float(row["steady_event_timing"]["mean_ms"])
                for row in condition_rows
            ],
            dtype=float,
        )
        axis.plot(
            vocabulary,
            update_ms,
            label=label,
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=1.6,
            markersize=5,
        )
        lower = [row.get("ci95_lower_ms") for row in condition_rows]
        upper = [row.get("ci95_upper_ms") for row in condition_rows]
        if timing_source == "window" and all(
            value is not None for value in (*lower, *upper)
        ):
            axis.fill_between(
                vocabulary,
                np.asarray(lower, dtype=float),
                np.asarray(upper, dtype=float),
                color=color,
                alpha=0.2,
                linewidth=0,
            )
    model_label = "Skip-gram" if model == "SkipGram" else model
    axis.set(
        xlabel="Vocabulary size",
        ylabel="Update time (ms)",
        xscale="log",
    )
    if title:
        axis.set_title(f"{model_label} · {device_label}")
    axis.grid(True, which="both", alpha=0.25, color=MUTED)
    axis.legend()
