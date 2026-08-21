"""DS1 GT06: reproduce the book's SimpleConvNet training graph."""

import matplotlib.pyplot as plt

from exp.framework.analysis.core import mark_empty, plot_curve

from .common import accuracy_percent_curve, runs


ATOMIC_RUN_ID = "CNN-SIMPLE-BOOK"


def render_cnn(client, error_style, *, group_id, atomic_run_id):
    grouped = runs(client, group_id, [atomic_run_id])
    curves = {
        split: accuracy_percent_curve(
            client,
            grouped[atomic_run_id],
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
        ylabel="accuracy (%)",
        ylim=(0, 100.0),
    )
    mark_empty(axis)
    if axis.has_data():
        axis.legend(loc="lower right")
    return figure, curves


def render(client, error_style, output):
    del output
    return render_cnn(
        client,
        error_style,
        group_id="GT06",
        atomic_run_id=ATOMIC_RUN_ID,
    )
