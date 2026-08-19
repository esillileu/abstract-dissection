"""DS2 GT05: compare validation perplexity on a broken y-axis."""

import matplotlib.pyplot as plt
import numpy as np

from exp.framework.analysis.core import aggregate, mark_empty, plot_curve
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


def _evaluation_curve(client, run_refs, *, split, axis):
    return aggregate(
        histories_from_artifact(
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
    )


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


def render(client, error_style, output):
    del output
    atomic_ids = [item[0] for item in DEFINITIONS]
    grouped = runs(client, "GT05", atomic_ids)
    curves = {
        f"{atomic}/valid": _evaluation_curve(
            client, grouped[atomic], split="valid", axis="epoch"
        )
        for atomic in atomic_ids
    }

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
