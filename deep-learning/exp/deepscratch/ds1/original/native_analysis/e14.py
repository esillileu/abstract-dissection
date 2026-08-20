"""Render the original ch05 gradient-check result."""

from pathlib import Path

from .common import load_csv, plt, save, trial


TRIAL_IDS = ("dlfs1.ch05.gradient-check",)


def render(root: Path, image_dir: Path) -> list[Path]:
    directory = trial(root, "e14", TRIAL_IDS[0])
    rows = load_csv(directory / "gradient_check.csv")
    metrics = load_csv(directory / "metrics.csv")
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(
        [row["parameter"] for row in rows],
        [float(row["mean_absolute_difference"]) for row in rows],
    )
    axes[0].set_yscale("log")
    axes[0].set(xlabel="parameter", ylabel="mean absolute difference")
    timing = {
        row["metric"].removeprefix("gradient_check/"): float(row["value"])
        for row in metrics
    }
    axes[1].bar(
        ("numerical", "backprop"),
        (timing["numerical_s"], timing["backprop_s"]),
    )
    axes[1].set_yscale("log")
    axes[1].set(ylabel="gradient computation time (s)")
    path = image_dir / "e14_gradient_check.png"
    save(path)
    return [path]
