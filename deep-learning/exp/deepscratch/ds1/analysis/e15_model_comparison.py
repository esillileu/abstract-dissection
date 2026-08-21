"""Compare new GT11 models with the reused GT07 DeepCNN run."""

import matplotlib.pyplot as plt

from exp.framework.analysis.core import mark_empty, plot_curve
from exp.framework.plotting.theme import ACCENT_COLORS

from .common import accuracy_percent_curve, runs


DEFINITIONS = (
    ("GT07", "CNN-DEEP-BOOK", "DeepCNN", ACCENT_COLORS[0]),
    ("GT11", "TWO-LAYER-NET-ADAM", "TwoLayerNet", ACCENT_COLORS[1]),
    ("GT11", "MLP-EXT-NO-REG", "ExtendedMLP-NoReg", ACCENT_COLORS[2]),
)


def _curves(client):
    grouped = {
        atomic_run_id: runs(client, group_id, [atomic_run_id])[atomic_run_id]
        for group_id, atomic_run_id, _label, _color in DEFINITIONS
    }
    return {
        f"{label}/{split}": accuracy_percent_curve(
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
    for _group_id, _atomic_run_id, label, color in DEFINITIONS:
        for split, marker, linestyle in (
            ("train", "o", "-"),
            ("test", "s", "--"),
        ):
            name = f"{label}/{split}"
            plot_curve(
                axis,
                curves[name],
                label=name,
                marker=marker,
                linestyle=linestyle,
                error_style=error_style,
                error_every=2,
                color=color,
            )
    mark_empty(axis)
    axis.set(xlabel="epochs", ylabel="accuracy (%)", ylim=(0.0, 100.0))
    if axis.has_data():
        axis.legend(loc="lower right")
    return figure, curves
