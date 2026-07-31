"""Render the original LSTM RNNLM training perplexity curve."""

from pathlib import Path

from .common import load_csv, np, plt, save, trial


TRIAL_IDS = ("dlfs2.ch06.ptb-lstm-rnnlm",)


def render(root: Path, image_dir: Path) -> list[Path]:
    rows = [
        row
        for row in load_csv(trial(root, "e04", TRIAL_IDS[0]) / "metrics.csv")
        if row["split"] == "train"
    ]
    plt.ylim(0, 500)
    plt.plot(
        np.arange(len(rows)),
        [float(row["perplexity"]) for row in rows],
        label="train",
    )
    plt.xlabel("iterations (x20)")
    plt.ylabel("perplexity")
    path = image_dir / "e04_lstm_rnnlm.png"
    save(path)
    return [path]
