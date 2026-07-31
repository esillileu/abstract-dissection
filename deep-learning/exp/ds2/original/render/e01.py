"""Render the original toy CBOW loss figure."""

from pathlib import Path

from .common import load_csv, np, plt, save, trial


TRIAL_IDS = ("dlfs2.ch03.toy-cbow-full-softmax",)


def render(root: Path, image_dir: Path) -> list[Path]:
    rows = load_csv(trial(root, "e01", TRIAL_IDS[0]) / "metrics.csv")
    plt.plot(np.arange(len(rows)), [float(row["loss"]) for row in rows], label="train")
    plt.xlabel("iterations (x20)")
    plt.ylabel("loss")
    path = image_dir / "e01_toy_cbow.png"
    save(path)
    return [path]
