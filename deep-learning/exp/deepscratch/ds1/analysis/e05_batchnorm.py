"""DS1 GT05: reproduce the 4x4 BatchNorm scale comparison."""

import matplotlib.pyplot as plt
import numpy as np

from exp.framework.analysis.core import mark_empty, plot_curve

from .common import accuracy_curve, runs


def render(client, error_style, output):
    del output
    atomic_ids = [f"BN-SCALE-{index:02d}-{state}" for index in range(1, 17) for state in ("ON", "OFF")]
    grouped = runs(client, "GT05", atomic_ids)
    figure, axes = plt.subplots(4, 4, figsize=(13, 10), sharex=True, sharey=True)
    figure._analysis_match_original_canvas = True
    figure.subplots_adjust(top=0.92, hspace=0.35, wspace=0.25)
    curves = {}
    for index, (axis, scale) in enumerate(zip(axes.flat, np.logspace(0, -4, 16), strict=True), start=1):
        for state, label, linestyle in (
            ("ON", "Batch Normalization", "-"),
            ("OFF", "Normal (without BatchNorm)", "--"),
        ):
            atomic = f"BN-SCALE-{index:02d}-{state}"
            curve = accuracy_curve(
                client,
                grouped[atomic],
                split="train",
                x_value=lambda row: float(row["epoch"]) - 1,
            )
            curves[atomic] = curve
            plot_curve(axis, curve, label=label, linestyle=linestyle, error_style=error_style, error_every=2)
        axis.set_title(f"W:{scale:g}")
        axis.set_ylim(0, 1)
        if index % 4 == 1:
            axis.set_ylabel("accuracy")
        if index > 12:
            axis.set_xlabel("epochs")
        mark_empty(axis)
    if axes.flat[0].has_data():
        axes.flat[0].legend(loc="lower right", fontsize=7)
    return figure, curves
