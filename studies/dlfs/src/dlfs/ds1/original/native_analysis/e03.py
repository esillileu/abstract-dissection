"""Render the original weight-decay overfitting figure."""

from pathlib import Path

from .common import load_csv, np, plt, save, trial

TRIAL_IDS = (
    "dlfs1.ch06.weight-decay.off",
    "dlfs1.ch06.weight-decay.lambda-01",
)


def render(root: Path, image_dir: Path) -> list[Path]:
    for trial_id, label, linestyle in (
        (TRIAL_IDS[0], "weight decay off", "-"),
        (TRIAL_IDS[1], "weight decay 0.1", "--"),
    ):
        rows = load_csv(trial(root, "e03", trial_id) / "metrics.csv")
        for split, marker in (("train", "o"), ("test", "s")):
            values = [float(row["accuracy"]) for row in rows if row["split"] == split]
            plt.plot(
                np.arange(len(values)),
                values,
                marker=marker,
                linestyle=linestyle,
                label=f"{label} {split}",
                markevery=10,
            )
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.ylim(0, 1.0)
    plt.legend(loc="lower right")
    path = image_dir / "e03_overfit_weight_decay.png"
    save(path)
    return [path]
