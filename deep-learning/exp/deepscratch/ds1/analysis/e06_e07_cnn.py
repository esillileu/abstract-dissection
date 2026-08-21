"""DS1 GT06+GT07: compare both CNNs on a shared broken y-axis."""

import matplotlib.pyplot as plt

from exp.framework.analysis.core import mark_empty, plot_curve
from exp.framework.plotting.theme import ACCENT_COLORS

from .broken_axis import add_wave_break
from .common import accuracy_percent_curve, runs


DEFINITIONS = [
    ("GT06", "CNN-SIMPLE-BOOK", "SimpleCNN", ACCENT_COLORS[0]),
    ("GT07", "CNN-DEEP-BOOK", "DeepCNN", ACCENT_COLORS[1]),
]


def _curves(client):
    grouped = {
        atomic: runs(client, group, [atomic])[atomic]
        for group, atomic, _label, _color in DEFINITIONS
    }
    result = {}
    for _group, atomic, label, _color in DEFINITIONS:
        for split in ("train", "test"):
            result[f"{label}/{split}"] = accuracy_percent_curve(
                client,
                grouped[atomic],
                split=split,
                axis="update",
                x_value=lambda row: float(row["epoch"]) - 1,
            )
    return result


def render_compare(client, error_style, output):
    del output
    curves = _curves(client)
    figure, (upper, lower) = plt.subplots(
        2,
        1,
        sharex=True,
        gridspec_kw={"height_ratios": (3, 1), "hspace": 0.05},
    )
    figure.subplots_adjust(left=0.125, right=0.9, bottom=0.11, top=0.88)
    figure._analysis_skip_tight_layout = True
    figure._analysis_match_original_canvas = True
    styles = {
        "SimpleCNN/train": ("o", "-", ACCENT_COLORS[0]),
        "SimpleCNN/test": ("s", "--", ACCENT_COLORS[0]),
        "DeepCNN/train": ("^", "-", ACCENT_COLORS[1]),
        "DeepCNN/test": ("D", "--", ACCENT_COLORS[1]),
    }
    for axis in (upper, lower):
        for name, curve in curves.items():
            marker, linestyle, color = styles[name]
            plot_curve(
                axis,
                curve,
                label=name,
                marker=marker,
                linestyle=linestyle,
                error_style=error_style,
                error_every=2,
                color=color,
            )
        mark_empty(axis)
    upper.set_ylim(95.0, 100.0)
    lower.set_ylim(0.0, 23.0)
    upper.set_yticks((96.0, 98.0, 100.0))
    lower.set_yticks((0.0, 10.0, 20.0))
    figure.text(0.02, 0.5, "accuracy (%)", va="center", rotation="vertical")
    lower.set_xlabel("epochs")
    add_wave_break(figure, upper, lower)
    if upper.has_data():
        upper.legend(loc="lower right")
    return figure, curves


__all__ = ["render_compare"]
