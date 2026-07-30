"""Render the four original addition Seq2seq conditions in one figure."""

from pathlib import Path

from .common import load_csv, np, plt, save, trial


SUFFIXES = (
    "seq2seq-forward",
    "seq2seq-reverse",
    "peeky-seq2seq-forward",
    "peeky-seq2seq-reverse",
)
TRIAL_IDS = tuple(f"dlfs2.ch07.addition.{suffix}" for suffix in SUFFIXES)
LABELS = (
    "vanilla / forward",
    "vanilla / reverse",
    "peeky / forward",
    "peeky / reverse",
)


def render(root: Path, image_dir: Path) -> list[Path]:
    for trial_id, label in zip(TRIAL_IDS, LABELS, strict=True):
        rows = load_csv(trial(root, "e06", trial_id) / "metrics.csv")
        plt.plot(
            np.arange(len(rows)),
            [float(row["accuracy"]) for row in rows],
            marker="o",
            label=label,
        )
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.ylim(0, 1.0)
    plt.title("Original Addition Seq2seq")
    plt.legend()
    path = image_dir / "e06_addition_seq2seq.png"
    save(path)
    return [path]
