"""Render the original optimizer-comparison figure."""

from pathlib import Path

from .common import floats, load_csv, np, plt, save, trial

TRIAL_IDS = tuple(
    f"dlfs1.ch06.optimizer-mnist.{name}"
    for name in ("sgd", "momentum", "adagrad", "adam")
)


def render(root: Path, image_dir: Path) -> list[Path]:
    markers = {"SGD": "o", "Momentum": "x", "AdaGrad": "s", "Adam": "D"}
    for trial_id, label in zip(
        TRIAL_IDS, ("SGD", "Momentum", "AdaGrad", "Adam"), strict=True
    ):
        values = floats(load_csv(trial(root, "e01", trial_id) / "metrics.csv"), "value")
        window_len = 11
        reflected = np.r_[
            values[window_len - 1 : 0 : -1], values, values[-1:-window_len:-1]
        ]
        weights = np.kaiser(window_len, 2)
        smoothed = np.convolve(weights / weights.sum(), reflected, mode="valid")
        smoothed = smoothed[5 : len(smoothed) - 5]
        plt.plot(
            np.arange(len(values)),
            smoothed,
            marker=markers[label],
            markevery=100,
            label=label,
        )
    plt.xlabel("iterations")
    plt.ylabel("loss")
    plt.ylim(0, 1)
    plt.legend()
    path = image_dir / "e01_optimizer_compare_mnist.png"
    save(path)
    return [path]
