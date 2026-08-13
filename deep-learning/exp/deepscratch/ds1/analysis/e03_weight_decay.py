"""DS1 GT03: reproduce train/test accuracy with weight decay."""

import matplotlib.pyplot as plt

from exp.analyze import mark_empty, plot_curve
from exp.plot_theme import ACCENT_COLORS, SECONDARY_DATA

from .common import accuracy_curve, runs


def render(client, error_style, output):
    del output
    definitions = [("REG-WD-OFF", "weight decay off"), ("REG-WD-01", "weight decay 0.1")]
    colors = {
        "REG-WD-OFF": {"train": SECONDARY_DATA, "test": SECONDARY_DATA},
        "REG-WD-01": {"train": ACCENT_COLORS[0], "test": ACCENT_COLORS[1]},
    }
    grouped = runs(client, "GT03", [item[0] for item in definitions])
    figure, axis = plt.subplots()
    figure._analysis_match_original_canvas = True
    curves = {}
    for atomic, label in definitions:
        for split, marker, linestyle in (("train", "o", "-"), ("test", "s", "--")):
            curve = accuracy_curve(
                client, grouped[atomic], split=split, x_value=lambda row: float(row["epoch"]) - 1
            )
            curves[f"{atomic}/{split}"] = curve
            plot_curve(
                axis,
                curve,
                label=f"{label} {split}",
                marker=marker,
                linestyle=linestyle,
                error_style=error_style,
                error_every=10,
                color=colors[atomic][split],
            )
    axis.set(xlabel="epochs", ylabel="accuracy", ylim=(0, 1))
    mark_empty(axis)
    if axis.has_data():
        axis.legend()
    return figure, curves
