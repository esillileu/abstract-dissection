"""DS1 GT08: compare identity and pixel-permuted NN/CNN accuracy."""

import matplotlib.pyplot as plt

from exp.analyze import aggregate, mark_empty, metric_histories, plot_curve

from .broken_axis import add_wave_break
from .common import runs


METRICS = {
    "test": "update/eval_test/accuracy",
    "train": "update/eval_train/accuracy",
}
PANELS = [
    (
        "ParameterMatchedNN",
        [
            ("NN-MATCHED", "NN", "o", "tab:blue"),
            ("NN-MATCHED-PERMUTED", "NN permuted", "s", "tab:orange"),
        ],
    ),
    (
        "SimpleConvNet",
        [
            ("CNN-SIMPLE-SPATIAL", "CNN", "o", "tab:blue"),
            ("CNN-SIMPLE-SPATIAL-PERMUTED", "CNN permuted", "s", "tab:orange"),
        ],
    ),
]


def render(client, error_style, output):
    del output
    atomic_ids = [definition[0] for _title, definitions in PANELS for definition in definitions]
    grouped = runs(client, "GT08", atomic_ids)
    curves = {
        f"{atomic}/{split}": aggregate(metric_histories(client, grouped[atomic], metric))
        for atomic in atomic_ids
        for split, metric in METRICS.items()
    }
    figure = plt.figure(figsize=(13, 6))
    grid = figure.add_gridspec(2, 2, height_ratios=(3, 1), hspace=0.05, wspace=0.16)
    figure.subplots_adjust(left=0.08, right=0.98, bottom=0.12, top=0.9)
    figure._analysis_skip_tight_layout = True
    panel_axes = []
    for column, (title, definitions) in enumerate(PANELS):
        upper = figure.add_subplot(grid[0, column])
        lower = figure.add_subplot(grid[1, column], sharex=upper)
        panel_axes.append((upper, lower))
        for axis in (upper, lower):
            for atomic, label, marker, color in definitions:
                for split, linestyle in (("test", "-"), ("train", ":")):
                    plot_curve(
                        axis,
                        curves[f"{atomic}/{split}"],
                        label=f"{label} {split}",
                        marker=marker if split == "test" else None,
                        linestyle=linestyle,
                        color=color,
                        error_style=error_style,
                        error_every=2,
                    )
            mark_empty(axis)
        upper.set_ylim(0.59, 1.00)
        lower.set_ylim(0.0, 0.25)

        # upper.set_yticks((0.7, 0.8, 0.94, 0.96, 0.98, 1.0))
        # lower.set_yticks((0.0, 0.1, 0.2, 0.25))
        upper.set_title(title)
        lower.set_xlabel("updates")
        add_wave_break(figure, upper, lower)
        if upper.has_data():
            upper.legend(loc="lower right")
    panel_axes[0][0].set_ylabel("accuracy")
    panel_axes[0][1].set_ylabel("accuracy")
    return figure, curves
