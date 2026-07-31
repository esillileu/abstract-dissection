"""Render the two original PTB Word2Vec loss figures."""

from pathlib import Path

from .common import load_csv, np, plt, save, trial


TRIAL_IDS = (
    "dlfs2.ch04.ptb-cbow-negative-sampling",
    "dlfs2.ch04.ptb-skipgram-negative-sampling",
)


def render(root: Path, image_dir: Path) -> list[Path]:
    outputs = []
    for trial_id, suffix in zip(TRIAL_IDS, ("cbow", "skipgram"), strict=True):
        rows = load_csv(trial(root, "e02", trial_id) / "metrics.csv")
        plt.plot(
            np.arange(len(rows)),
            [float(row["loss"]) for row in rows],
            label="train",
        )
        plt.xlabel("iterations (x20)")
        plt.ylabel("loss")
        path = image_dir / f"e02_ptb_{suffix}.png"
        save(path)
        outputs.append(path)
    return outputs
