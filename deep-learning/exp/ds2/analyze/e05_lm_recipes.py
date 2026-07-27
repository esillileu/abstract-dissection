"""DS2 GT05: compare validation perplexity on a broken y-axis."""

import matplotlib.pyplot as plt
import numpy as np

from exp.analyze import aggregate, histories_from_artifact, mark_empty, plot_curve
from exp.plot_theme import ACCENT_COLORS

from .broken_axis import add_wave_break
from .common import runs


DEFINITIONS = [
    ("LM-RNN-RECIPE", "Vanilla RNNLM", "o", ACCENT_COLORS[0]),
    ("LM-LSTM-RECIPE", "LSTM RNNLM", "s", ACCENT_COLORS[1]),
    ("LM-BETTER-RECIPE", "BetterRNNLM", "D", ACCENT_COLORS[2]),
]


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
        gridspec_kw={"height_ratios": (1, 3), "hspace": 0.05},
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
    upper_extrema = _finite_extrema([curves[f"{atomic_ids[0]}/valid"]])
    upper_limits = _padded_limits(upper_extrema)
    if lower_limits is not None:
        lower.set_ylim(*lower_limits)
    if upper_limits is not None and upper_extrema is not None:
        upper.set_ylim(max(1.0, upper_extrema[0] * 0.5), upper_limits[1])
        upper.ticklabel_format(axis="y", style="plain", useOffset=False)

    upper.set_title("Validation perplexity")
    lower.set_xlabel("epochs")
    figure.text(0.025, 0.5, "perplexity", va="center", rotation="vertical")
    add_wave_break(figure, upper, lower)
    if lines:
        upper.legend(handles=lines, loc="upper right")
    return figure, curves
