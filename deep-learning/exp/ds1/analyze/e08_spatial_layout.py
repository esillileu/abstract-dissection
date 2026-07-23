"""DS1 GT08: compare identity and pixel-permuted NN/CNN accuracy."""

import matplotlib.pyplot as plt
import numpy as np

from mlprosection.datasets.mnist import load_mnist

from exp.analyze import aggregate, mark_empty, metric_histories, plot_curve, save_figure

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


def _permute_image(image: np.ndarray, *, seed: int) -> np.ndarray:
    permutation = np.random.default_rng(seed).permutation(image.size)
    return image.reshape(-1)[permutation].reshape(image.shape)


def _render_permutation_example(output) -> None:
    (_x_train, _t_train), (x_test, t_test) = load_mnist(flatten=False)
    original = np.asarray(x_test[PERMUTATION_EXAMPLE_INDEX])
    permuted = _permute_image(original, seed=PIXEL_PERMUTATION_SEED)

    figure, axes = plt.subplots(1, 2, figsize=(7, 3.5))
    for axis, image, title in zip(
        axes,
        (original, permuted),
        ("Original", "Pixel-permuted"),
        strict=True,
    ):
        axis.imshow(
            image.squeeze(),
            cmap="gray",
            interpolation="nearest",
            vmin=0.0,
            vmax=1.0,
        )
        axis.set_title(title)
        axis.set_xticks(())
        axis.set_yticks(())
    figure.suptitle(
        f"MNIST test sample {PERMUTATION_EXAMPLE_INDEX} | "
        f"label {int(t_test[PERMUTATION_EXAMPLE_INDEX])} | "
        f"permutation seed {PIXEL_PERMUTATION_SEED}"
    )
    save_figure(figure, output)
    plt.close(figure)


def render(client, error_style, output):
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
    example_output = output.with_name(f"{output.stem}_permutation-example.png")
    _render_permutation_example(example_output)
    figure._analysis_extra_outputs = [example_output]
    return figure, curves
