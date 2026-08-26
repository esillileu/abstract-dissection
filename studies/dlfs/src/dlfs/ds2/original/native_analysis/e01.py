"""Render the original toy CBOW and Skip-gram loss figures."""

from pathlib import Path

from .common import load_csv, np, plt, save, trial

TRIAL_IDS = (
    "dlfs2.ch03.toy-cbow-full-softmax",
    "dlfs2.ch03.toy-skipgram-full-softmax",
)


def render(root: Path, image_dir: Path) -> list[Path]:
    outputs = []
    for trial_id, name in zip(TRIAL_IDS, ("cbow", "skipgram"), strict=True):
        rows = load_csv(trial(root, "e01", trial_id) / "metrics.csv")
        plt.plot(
            np.arange(len(rows)),
            [float(row["loss"]) for row in rows],
            label="train",
        )
        plt.xlabel("iterations (x20)")
        plt.ylabel("loss")
        path = image_dir / f"e01_toy_{name}.png"
        save(path)
        outputs.append(path)
    return outputs
