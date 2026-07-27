"""Render the original small RNNLM perplexity curve."""

from pathlib import Path

from .common import load_csv, np, plt, save, trial


TRIAL_IDS = ("dlfs2.ch05.ptb-small-rnnlm",)


def render(root: Path, image_dir: Path) -> list[Path]:
    rows = load_csv(trial(root, "e03", TRIAL_IDS[0]) / "metrics.csv")
    plt.plot(
        np.arange(len(rows)),
        [float(row["perplexity"]) for row in rows],
        label="train",
    )
    plt.xlabel("iterations (x20)")
    plt.ylabel("perplexity")
    plt.title("Original PTB small-corpus SimpleRnnlm")
    path = image_dir / "e03_small_rnnlm.png"
    save(path)
    return [path]
