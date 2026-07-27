"""Render the original SimpleConvNet learning curve."""

from pathlib import Path

from .common import load_csv, np, plt, save, trial


TRIAL_IDS = ("dlfs1.ch07.simple-convnet",)


def render(root: Path, image_dir: Path) -> list[Path]:
    rows = load_csv(trial(root, "e06", TRIAL_IDS[0]) / "metrics.csv")
    for split, marker in (("train", "o"), ("test", "s")):
        values = [
            float(row["accuracy"])
            for row in rows
            if row["split"] == split and int(row["epoch"]) < 20
        ]
        plt.plot(
            np.arange(20), values, marker=marker, label=split, markevery=2
        )
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.ylim(0, 1.0)
    plt.title("Original SimpleConvNet vs DeepConvNet")
    plt.legend(loc="lower right")
    path = image_dir / "e06_train_convnet.png"
    save(path)
    return [path]
