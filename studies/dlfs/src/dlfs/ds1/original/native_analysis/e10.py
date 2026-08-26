"""Render the original five activation histograms."""

from pathlib import Path

from .common import load_npz, plt, save, trial

TRIAL_IDS = ("dlfs1.ch06.activation.sigmoid-std-1",)


def render(root: Path, image_dir: Path) -> list[Path]:
    arrays = load_npz(trial(root, "e10", TRIAL_IDS[0]) / "activations.npz")
    figure = plt.figure()
    figure.suptitle("Original ACT-SIGMOID-STD1")
    for index in range(5):
        values = arrays[f"layer_{index + 1}"]
        figure.add_subplot(1, 5, index + 1)
        plt.title(f"{index + 1}-layer")
        if index != 0:
            plt.yticks([], [])
        plt.hist(values.flatten(), 30, range=(0, 1))
    path = image_dir / "e10_weight_init_activation_histogram.png"
    save(path)
    return [path]
