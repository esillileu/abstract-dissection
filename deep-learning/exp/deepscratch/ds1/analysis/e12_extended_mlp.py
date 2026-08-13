"""Compare the book-option extended MLP with the existing DeepConvNet runs."""

import matplotlib.pyplot as plt

from exp.framework.analysis.core import mark_empty, plot_curve
from exp.framework.plotting.theme import ACCENT_COLORS

from .common import accuracy_curve, runs


DEFINITIONS = (
    ("GT07", "CNN-DEEP-BOOK", "DeepCNN", ACCENT_COLORS[0]),
    ("GT09", "MLP-EXT-ALL-BOOK", "ExtendedMLP", ACCENT_COLORS[1]),
)


def _curves(client):
    grouped = {
        atomic_run_id: runs(client, group_id, [atomic_run_id])[atomic_run_id]
        for group_id, atomic_run_id, _label, _color in DEFINITIONS
    }
    return {
        f"{label}/{split}": accuracy_curve(
            client,
            grouped[atomic_run_id],
            split=split,
            axis="update",
            x_value=lambda row: float(row["epoch"]) - 1,
        )
        for _group_id, atomic_run_id, label, _color in DEFINITIONS
        for split in ("train", "test")
    }


def render(client, error_style, output):
    del output
    curves = _curves(client)
    figure, axis = plt.subplots()
    styles = {
        "DeepCNN/train": ("o", "-", ACCENT_COLORS[0]),
        "DeepCNN/test": ("s", "--", ACCENT_COLORS[0]),
        "ExtendedMLP/train": ("^", "-", ACCENT_COLORS[1]),
        "ExtendedMLP/test": ("D", "--", ACCENT_COLORS[1]),
    }
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
    axis.set(xlabel="epochs", ylabel="accuracy", ylim=(0.0, 1.0))
    if axis.has_data():
        axis.legend(loc="lower right")
    return figure, curves
