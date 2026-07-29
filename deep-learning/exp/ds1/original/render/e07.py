"""Render the original DeepConvNet learning curve."""

from pathlib import Path

from .common import load_csv, np, plt, save, trial


TRIAL_IDS = ("dlfs1.ch08.deep-convnet",)


def render(root: Path, image_dir: Path) -> list[Path]:
    rows = load_csv(trial(root, "e07", TRIAL_IDS[0]) / "metrics.csv")
    for split, marker in (("train", "o"), ("test", "s")):
        values = [
            float(row["accuracy"])
            for row in rows
            if row["split"] == split and int(row["epoch"]) < 20
        ]
        plt.plot(
            np.arange(20),
            values,
            marker=marker,
            label=split,
            markevery=2,
        )
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.ylim(0, 1.0)
    plt.title("Original DeepConvNet Training")
    plt.legend(loc="lower right")
    path = image_dir / "e07_train_deepnet.png"
    save(path)
    return [path]
