"""Render the book's backprop-trained two-layer net."""

import matplotlib.pyplot as plt

from repro_core.analysis.core import mark_empty, plot_curve

from .common import accuracy_curve, runs

ATOMIC_RUN_ID = "TWO-LAYER-NET-BACKPROP"


def render(client, error_style, output):
    del output
    grouped = runs(client, "GT10", [ATOMIC_RUN_ID])
    curves = {
        split: accuracy_curve(
            client,
            grouped[ATOMIC_RUN_ID],
            split=split,
            axis="update",
            x_value=lambda row: float(row["epoch"]) - 1,
        )
        for split in ("train", "test")
    }

    figure, axis = plt.subplots()
    figure._analysis_match_original_canvas = True
    for split, marker in (("train", "o"), ("test", "s")):
        plot_curve(
            axis,
            curves[split],
            label=split,
            marker=marker,
            error_style=error_style,
            error_every=2,
        )
    axis.set(
        xlabel="epochs",
        ylabel="accuracy",
        ylim=(0.0, 1.0),
    )
    mark_empty(axis)
    if axis.has_data():
        axis.legend(loc="lower right")
    return figure, curves
