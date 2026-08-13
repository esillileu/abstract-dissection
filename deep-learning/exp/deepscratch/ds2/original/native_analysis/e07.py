"""Render the three original date Seq2seq conditions in one figure."""

from pathlib import Path

from .common import load_csv, np, plt, save, trial


SUFFIXES = (
    "seq2seq-reverse",
    "peeky-seq2seq-reverse",
    "attention-seq2seq-reverse",
)
TRIAL_IDS = tuple(f"dlfs2.ch08.date.{suffix}" for suffix in SUFFIXES)
LABELS = ("vanilla", "peeky", "attention")


def render(root: Path, image_dir: Path) -> list[Path]:
    for trial_id, label in zip(TRIAL_IDS, LABELS, strict=True):
        rows = load_csv(trial(root, "e07", trial_id) / "metrics.csv")
        plt.plot(
            np.arange(len(rows)),
            [float(row["accuracy"]) for row in rows],
            marker="o",
            label=label,
        )
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.ylim(-0.05, 1.05)
    plt.legend()
    path = image_dir / "e07_date_seq2seq.png"
    save(path)
    return [path]
