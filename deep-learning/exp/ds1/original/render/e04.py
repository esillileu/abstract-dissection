"""Render the original dropout overfitting figure."""

from pathlib import Path

from .common import load_csv, np, plt, save, trial


TRIAL_IDS = ("dlfs1.ch06.dropout.on-ratio-02",)


def render(root: Path, image_dir: Path) -> list[Path]:
    rows = load_csv(trial(root, "e04", TRIAL_IDS[0]) / "metrics.csv")
    for split, marker in (("train", "o"), ("test", "s")):
        values = [float(row["accuracy"]) for row in rows if row["split"] == split]
        plt.plot(
            np.arange(len(values)),
            values,
            marker=marker,
            label=split,
            markevery=10,
        )
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.ylim(0, 1.0)
    plt.legend(loc="lower right")
    path = image_dir / "e04_overfit_dropout.png"
    save(path)
    return [path]
