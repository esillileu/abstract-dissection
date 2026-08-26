"""Render the original initialization-comparison figure."""

from pathlib import Path

from .common import floats, load_csv, np, plt, save, trial

TRIAL_IDS = (
    "dlfs1.ch06.init-compare.std-001",
    "dlfs1.ch06.init-compare.xavier",
    "dlfs1.ch06.init-compare.he",
)


def render(root: Path, image_dir: Path) -> list[Path]:
    markers = {"std=0.01": "o", "Xavier": "s", "He": "D"}
    for trial_id, label in zip(TRIAL_IDS, ("std=0.01", "Xavier", "He"), strict=True):
        values = floats(load_csv(trial(root, "e02", trial_id) / "metrics.csv"), "value")
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
    plt.ylim(0, 2.5)
    plt.legend()
    path = image_dir / "e02_weight_init_compare.png"
    save(path)
    return [path]
