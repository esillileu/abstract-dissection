"""DS1 GO02: reproduce one five-layer activation histogram per condition."""

import matplotlib.pyplot as plt
import numpy as np

from exp.framework.analysis.core import aggregate, mark_empty, save_figure, write_summary
from exp.deepscratch.analysis.input import histories_from_artifact

from .common import runs


ATOMIC_IDS = [
    f"ACT-{activation}-{initializer}"
    for activation in ("SIGMOID", "TANH", "RELU")
    for initializer in ("STD001", "XAVIER", "HE", "STD1")
]


def _curve(client, run_refs, layer):
    return aggregate(
        histories_from_artifact(
            client,
            run_refs,
            artifact_path="observations/activation_histogram.csv",
            x="bin_index",
            y="count",
            row_filter=lambda row: int(row["layer"]) == layer,
            x_value=lambda row: (float(row["bin_left"]) + float(row["bin_right"])) / 2,
        )
    )


def render(client, error_style, output):
    grouped = runs(client, "GO02", ATOMIC_IDS)
    outputs = []
    curves = {}
    for atomic in ATOMIC_IDS:
        condition_output = output.with_name(f"{output.stem}_{atomic.lower()}{output.suffix}")
        figure, axes = plt.subplots(1, 5, figsize=(15, 3.2), sharey=True)
        for layer, axis in enumerate(axes, start=1):
            curve = _curve(client, grouped[atomic], layer)
            curves[f"{atomic}/layer-{layer}"] = curve
            if len(curve.steps):
                line = axis.step(curve.steps, curve.mean, where="mid", label=f"mean (n={curve.run_count})")[0]
                if error_style == "band":
                    axis.fill_between(
                        curve.steps,
                        curve.minimum,
                        curve.maximum,
                        step="mid",
                        alpha=0.2,
                        color=line.get_color(),
                    )
                else:
                    axis.errorbar(
                        curve.steps,
                        curve.mean,
                        yerr=np.vstack((curve.mean - curve.minimum, curve.maximum - curve.mean)),
                        fmt="none",
                        ecolor=line.get_color(),
                        elinewidth=0.7,
                        capsize=1,
                    )
            axis.set_title(f"{layer}-layer")
            axis.set_xlabel("activation")
            if layer == 1:
                axis.set_ylabel("count")
            mark_empty(axis)
        figure.suptitle(atomic)
        if grouped[atomic]:
            save_figure(figure, condition_output)
            outputs.append(condition_output)
        plt.close(figure)
    if not outputs:
        figure, axis = plt.subplots(figsize=(8, 4))
        mark_empty(axis)
        save_figure(figure, output)
        plt.close(figure)
        outputs.append(output)
    summary = output.with_suffix(".csv")
    write_summary(summary, curves)
    outputs.append(summary)
    return outputs
