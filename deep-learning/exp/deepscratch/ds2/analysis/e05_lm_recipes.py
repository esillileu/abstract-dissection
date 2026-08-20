"""DS2 GT05: compare validation perplexity on a broken y-axis."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from exp.framework.analysis.core import Curve, aggregate, mark_empty, plot_curve, save_figure
from exp.deepscratch.analysis.input import histories_from_artifact
from exp.framework.plotting.theme import ACCENT_COLORS

from .broken_axis import add_wave_break
from .common import runs


DEFINITIONS = [
    ("LM-RNN-RECIPE", "Vanilla RNNLM", "o", ACCENT_COLORS[0]),
    ("LM-LSTM-RECIPE", "LSTM RNNLM", "s", ACCENT_COLORS[1]),
    ("LM-LSTM-TIED-RECIPE", "Tied RNNLM", "^", ACCENT_COLORS[2]),
    ("LM-BETTER-RECIPE", "Better RNNLM", "D", ACCENT_COLORS[3]),
    ("LM-BETTER-NODROPOUT", "Better RNNLM (no dropout)", "X", ACCENT_COLORS[4]),
]

# Manually choose the visible Vanilla RNNLM range here.
UPPER_Y_LIMITS = (100, 10000000)
UPPER_LOG_LINEAR_THRESHOLD = 250
# Increase the first value to move the wave break downward.
PANEL_HEIGHT_RATIOS = (2, 3)

ADDITIONAL_RNNLM_GRAPH = (
    ("LM-RNN-RECIPE", "RNN RNNLM", "o", ACCENT_COLORS[0]),
    ("LM-LSTM-RECIPE", "LSTM RNNLM", "s", ACCENT_COLORS[1]),
)
ADDITIONAL_BETTER_GRAPH = (
    ("LM-BETTER-RECIPE", "Better RNNLM", "D", ACCENT_COLORS[3]),
    ("LM-BETTER-NODROPOUT", "Better RNNLM (no dropout)", "X", ACCENT_COLORS[4]),
)
ADDITIONAL_LSTM_GRAPH = (
    ("LM-LSTM-RECIPE", "RNNLM", "s", ACCENT_COLORS[1]),
    ("LM-LSTM-TIED-RECIPE", "RNNLM(weight tying)", "^", ACCENT_COLORS[2]),
)


def _evaluation_curve(client, run_refs, *, split, axis):
    histories = histories_from_artifact(
            client,
            run_refs,
            artifact_path="evaluations.csv",
            x="axis_step",
            y="value",
            row_filter=lambda row: row.get("axis") == axis
            and row.get("split") == split
            and row.get("metric") == "perplexity",
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
                {
                    epoch: sum(values) / len(values)
                    for epoch, values in by_epoch.items()
                }
                for run in grouped[atomic]
                for by_epoch in [_train_ppl_by_epoch(client, run)]
                if by_epoch
            ]
        )
        for atomic in atomic_ids
    }


def _train_ppl_by_epoch(client, run):
    by_epoch = {}
    for row in client.artifact_rows(run, "observations/source_curves.csv"):
        if row.get("metric") != "perplexity":
            continue
        try:
            epoch = float(row["epoch_end"])
            value = float(row["value"])
        except (KeyError, TypeError, ValueError):
            continue
        by_epoch.setdefault(epoch, []).append(value)
    return by_epoch


def render(client, error_style, output):
    del output
    atomic_ids = [item[0] for item in DEFINITIONS]
    curves = _valid_curves(client, atomic_ids)

    figure, (upper, lower) = plt.subplots(
        2,
        1,
        figsize=(9, 6),
        sharex=True,
        gridspec_kw={
            "height_ratios": PANEL_HEIGHT_RATIOS,
            "hspace": 0.05,
        },
    )
    figure.subplots_adjust(left=0.12, right=0.98, bottom=0.12, top=0.91)
    figure._analysis_skip_tight_layout = True
    lines = []
    for index, (atomic, label, marker, color) in enumerate(DEFINITIONS):
        axis = upper if index == 0 else lower
        line = plot_curve(
            axis,
            curves[f"{atomic}/valid"],
            label=label,
            marker=marker,
            color=color,
            error_style=error_style,
            error_every=1,
        )
        if line is not None:
            lines.append(line)
    for axis in (upper, lower):
        mark_empty(axis)
    lower_limits = _padded_limits(
        _finite_extrema([curves[f"{atomic}/valid"] for atomic in atomic_ids[1:]])
    )
    if lower_limits is not None:
        lower.set_ylim(*lower_limits)
    upper.set_yscale("symlog", linthresh=UPPER_LOG_LINEAR_THRESHOLD)
    upper.set_ylim(*UPPER_Y_LIMITS)

    lower.set_xlabel("epochs")
    figure.text(0.025, 0.5, "perplexity", va="center", rotation="vertical")
    add_wave_break(figure, upper, lower)
    if lines:
        upper.legend(handles=lines, loc="upper right")
    return figure, curves


def _render_single_axis_graph(
    client,
    error_style,
    output,
    definitions,
    filename,
    *,
    include_train=False,
    evaluation_split="valid",
    evaluation_axis="epoch",
    y_min=None,
    y_max=None,
) -> Path:
    atomic_ids = [item[0] for item in definitions]
    if evaluation_split == "valid" and evaluation_axis == "epoch":
        curves = _valid_curves(client, atomic_ids)
    else:
        curves = _evaluation_curves(
            client, atomic_ids, split=evaluation_split, axis=evaluation_axis
        )
    train_curves = _train_curves(client, atomic_ids) if include_train else {}
    figure, axis = plt.subplots()
    for atomic, label, marker, color in definitions:
        evaluation_curve = curves[f"{atomic}/valid"]
        if evaluation_axis == "terminal" and include_train and len(evaluation_curve.steps):
            final_step = train_curves[f"{atomic}/train"].steps.max()
            evaluation_curve = Curve(
                steps=np.full_like(evaluation_curve.steps, final_step),
                mean=evaluation_curve.mean,
                minimum=evaluation_curve.minimum,
                maximum=evaluation_curve.maximum,
                run_count=evaluation_curve.run_count,
                standard_deviation=evaluation_curve.standard_deviation,
            )
        plot_curve(
            axis,
            evaluation_curve,
            label=label,
            marker=marker,
            color=color,
            error_style=error_style,
            error_every=1,
        )
        if include_train:
            plot_curve(
                axis,
                train_curves[f"{atomic}/train"],
                label=f"{label} (train)",
                color=color,
                linestyle=":",
                error_style=error_style,
                error_every=1,
            )
    mark_empty(axis)
    axis.set(xlabel="epochs", ylabel="perplexity")
    if y_min is not None or y_max is not None:
        axis.set_ylim(bottom=y_min, top=y_max)
    if axis.has_data():
        axis.legend(loc="upper right")
    path = Path(output).with_name(filename)
    save_figure(figure, path)
    plt.close(figure)
    return path


def _render_broken_axis_graph(client, error_style, output, definitions, filename) -> Path:
    atomic_ids = [item[0] for item in definitions]
    curves = _valid_curves(client, atomic_ids)
    figure, (upper, lower) = plt.subplots(
        2,
        1,
        figsize=(6.4, 4.8),
        sharex=True,
        gridspec_kw={"height_ratios": PANEL_HEIGHT_RATIOS, "hspace": 0.05},
    )
    figure.subplots_adjust(left=0.12, right=0.98, bottom=0.12, top=0.91)
    figure._analysis_skip_tight_layout = True
    lines = []
    for index, (atomic, label, marker, color) in enumerate(definitions):
        axis = upper if index == 0 else lower
        line = plot_curve(
            axis,
            curves[f"{atomic}/valid"],
            label=label,
            marker=marker,
            color=color,
            error_style=error_style,
            error_every=1,
        )
        if line is not None:
            lines.append(line)
    for axis in (upper, lower):
        mark_empty(axis)
    lower_limits = _padded_limits(
        _finite_extrema([curves[f"{atomic}/valid"] for atomic in atomic_ids[1:]])
    )
    if lower_limits is not None:
        lower.set_ylim(*lower_limits)
    upper.set_yscale("symlog", linthresh=UPPER_LOG_LINEAR_THRESHOLD)
    upper.set_ylim(*UPPER_Y_LIMITS)
    lower.set_xlabel("epochs")
    figure.text(0.025, 0.5, "perplexity", va="center", rotation="vertical")
    add_wave_break(figure, upper, lower)
    if lines:
        upper.legend(handles=lines, loc="upper right")
    path = Path(output).with_name(filename)
    save_figure(figure, path)
    plt.close(figure)
    return path


def render_additional_graph(client, error_style, output) -> Path:
    """Render a focused validation-PPL comparison of the two RNNLM models."""
    return _render_broken_axis_graph(
        client,
        error_style,
        output,
        ADDITIONAL_RNNLM_GRAPH,
        "ds2_e05_rnn_lstm.png",
    )


def render_additional_better_graph(client, error_style, output) -> Path:
    """Render Better RNNLM with and without dropout on one regular axis."""
    return _render_single_axis_graph(
        client,
        error_style,
        output,
        ADDITIONAL_BETTER_GRAPH,
        "ds2_e05_better_rnnlm_dropout.png",
        include_train=True,
        evaluation_split="valid",
        evaluation_axis="epoch",
        y_min=0,
        y_max=500,
    )


def render_additional_lstm_graph(client, error_style, output) -> Path:
    """Render validation PPL for LSTM and tied LSTM RNNLMs."""
    return _render_single_axis_graph(
        client,
        error_style,
        output,
        ADDITIONAL_LSTM_GRAPH,
        "ds2_e05_lstm_vs_tied_rnnlm.png",
    )
