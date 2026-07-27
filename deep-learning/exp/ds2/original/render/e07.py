"""Render the three original date Seq2seq condition figures."""

from pathlib import Path

from .common import load_csv, np, plt, save, trial


SUFFIXES = (
    "seq2seq-reverse",
    "peeky-seq2seq-reverse",
    "attention-seq2seq-reverse",
)
TRIAL_IDS = tuple(f"dlfs2.ch08.date.{suffix}" for suffix in SUFFIXES)


def render(root: Path, image_dir: Path) -> list[Path]:
    outputs = []
    for trial_id, suffix in zip(TRIAL_IDS, SUFFIXES, strict=True):
        rows = load_csv(trial(root, "e07", trial_id) / "metrics.csv")
        plt.plot(
            np.arange(len(rows)),
            [float(row["accuracy"]) for row in rows],
            marker="o",
        )
        plt.xlabel("epochs")
        plt.ylabel("accuracy")
        plt.ylim(-0.05, 1.05)
        plt.title("Original Date Seq2seq")
        path = image_dir / f"e07_{suffix}.png"
        save(path)
        outputs.append(path)
    return outputs
