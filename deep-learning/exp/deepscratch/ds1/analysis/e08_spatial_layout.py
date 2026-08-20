"""DS1 GT08: compare identity and pixel-permuted NN/CNN accuracy."""

import matplotlib.pyplot as plt
import numpy as np

from mlprosection.datasets.mnist import load_mnist

from exp.framework.analysis.core import (
    aggregate,
    mark_empty,
    plot_curve,
    save_figure,
    write_summary,
)
from exp.deepscratch.analysis.input import metric_histories
from exp.framework.plotting.theme import ACCENT_COLORS

from .broken_axis import add_wave_break
from .common import runs


METRICS = {
    "test": "update/eval_test/accuracy",
    "train": "update/eval_train/accuracy",
}
PIXEL_PERMUTATION_SEED = 20260808
PERMUTATION_EXAMPLE_INDEX = 0
PANELS = [
    (
        "ParameterMatchedNN",
        [
            ("NN-MATCHED", "Normal", "o", ACCENT_COLORS[0]),
            ("NN-MATCHED-PERMUTED", "Permutated", "s", ACCENT_COLORS[1]),
        ],
    ),
    (
        "SimpleConvNet",
        [
            ("CNN-SIMPLE-SPATIAL", "Normal", "o", ACCENT_COLORS[0]),
            ("CNN-SIMPLE-SPATIAL-PERMUTED", "Permutated", "s", ACCENT_COLORS[1]),
        ],
    ),
]
PANEL_OUTPUT_NAMES = {
    "ParameterMatchedNN": "parameter_matched_nn",
    "SimpleConvNet": "simple_conv_net",
}


def _permute_image(image: np.ndarray, *, seed: int) -> np.ndarray:
    permutation = np.random.default_rng(seed).permutation(image.size)
    return image.reshape(-1)[permutation].reshape(image.shape)


def _add_permutation_examples(
    axis,
    normal: np.ndarray,
    permutated: np.ndarray,
) -> None:
    for bounds, image in zip(
        ((0.37, 1.05, 0.14, 0.50), (0.37, 0.45, 0.14, 0.50)),
        (normal, permutated),
        strict=True,
    ):
        image_axis = axis.inset_axes(bounds)
        image_axis.set_zorder(30)
        image_axis.imshow(
            image.squeeze(),
            cmap="gray",
            interpolation="nearest",
            vmin=0.0,
            vmax=1.0,
        )
        image_axis.set_xticks(())
        image_axis.set_yticks(())


def render(client, error_style, output):
    atomic_ids = [definition[0] for _title, definitions in PANELS for definition in definitions]
    grouped = runs(client, "GT08", atomic_ids)
    curves = {
        f"{atomic}/{split}": aggregate(metric_histories(client, grouped[atomic], metric))
        for atomic in atomic_ids
        for split, metric in METRICS.items()
    }
    (_x_train, _t_train), (x_test, _t_test) = load_mnist(flatten=False)
    normal = np.asarray(x_test[PERMUTATION_EXAMPLE_INDEX])
    permutated = _permute_image(normal, seed=PIXEL_PERMUTATION_SEED)

    outputs = []
    for panel_name, definitions in PANELS:
        figure = plt.figure(figsize=(6.5, 6))
        grid = figure.add_gridspec(2, 1, height_ratios=(3, 1), hspace=0.05)
        figure.subplots_adjust(left=0.12, right=0.98, bottom=0.12, top=0.98)
        figure._analysis_skip_tight_layout = True
        upper = figure.add_subplot(grid[0, 0])
        lower = figure.add_subplot(grid[1, 0], sharex=upper)
        lower.set_zorder(30)
        lower.patch.set_visible(False)
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
        lower.set_xlabel("updates")
        add_wave_break(figure, upper, lower)
        _add_permutation_examples(lower, normal, permutated)
        if upper.has_data():
            lower.legend(
                loc="center left",
                bbox_to_anchor=(0.48, 1.0),
                borderpad=0.8,
                fontsize=11,
                framealpha=1.0,
                handlelength=3.0,
                labelspacing=0.7,
            ).set_zorder(30)
        upper.set_ylabel("accuracy")
        lower.set_ylabel("accuracy")
        panel_output = output.with_name(
            f"{output.stem}_{PANEL_OUTPUT_NAMES[panel_name]}{output.suffix}"
        )
        save_figure(figure, panel_output)
        outputs.append(panel_output)
        plt.close(figure)
    summary_output = output.with_suffix(".csv")
    write_summary(summary_output, curves)
    outputs.append(summary_output)
    return outputs
