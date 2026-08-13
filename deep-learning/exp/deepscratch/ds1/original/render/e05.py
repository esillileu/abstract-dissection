"""Render the original 4x4 batch-normalization sweep."""

from pathlib import Path

from .common import load_csv, np, plt, save, trial


TRIAL_IDS = tuple(
    f"dlfs1.ch06.batchnorm.scale-{index:02d}.{suffix}"
    for index in range(1, 17)
    for suffix in ("bn-off", "bn-on")
)


def render(root: Path, image_dir: Path) -> list[Path]:
    scales = np.logspace(0, -4, num=16)
    x = np.arange(20)
    figure = plt.figure(figsize=(13, 10))
    figure.subplots_adjust(top=0.92, hspace=0.35, wspace=0.25)
    figure.suptitle("Original BatchNorm scale comparison")
    for index, scale in enumerate(scales):
        normal = load_csv(
            trial(
                root,
                "e05",
                f"dlfs1.ch06.batchnorm.scale-{index + 1:02d}.bn-off",
            )
            / "metrics.csv"
        )
        batchnorm = load_csv(
            trial(
                root,
                "e05",
                f"dlfs1.ch06.batchnorm.scale-{index + 1:02d}.bn-on",
            )
            / "metrics.csv"
        )
        figure.add_subplot(4, 4, index + 1)
        plt.title("W:" + str(scale))
        bn_values = [float(row["accuracy"]) for row in batchnorm]
        normal_values = [float(row["accuracy"]) for row in normal]
        if index == 0:
            plt.plot(x, bn_values, label="Batch Normalization", markevery=2)
            plt.plot(
                x,
                normal_values,
                linestyle="--",
                label="Normal(without BatchNorm)",
                markevery=2,
            )
        else:
            plt.plot(x, bn_values, markevery=2)
            plt.plot(x, normal_values, linestyle="--", markevery=2)
        plt.ylim(0, 1.0)
        if index % 4:
            plt.yticks([])
        else:
            plt.ylabel("accuracy")
        if index < 12:
            plt.xticks([])
        else:
            plt.xlabel("epochs")
        if index == 0:
            plt.legend(loc="lower right", fontsize=7)
    path = image_dir / "e05_batch_norm_test.png"
    save(path)
    return [path]
