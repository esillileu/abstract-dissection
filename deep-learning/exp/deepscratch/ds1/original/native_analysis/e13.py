"""Render original two-layer-net train/test accuracy curves."""

from pathlib import Path

from .common import load_csv, np, plt, save, trial


TRIAL_IDS = (
    "dlfs1.ch05.two-layer-net.backprop",
)


def render(root: Path, image_dir: Path) -> list[Path]:
    labels = ("Backpropagation",)
    markers = (("o", "s"),)
    for trial_id, label, marker in zip(TRIAL_IDS, labels, markers, strict=True):
        rows = load_csv(trial(root, "e13", trial_id) / "metrics.csv")
        for split, point, linestyle in (("train", marker[0], "-"), ("test", marker[1], "--")):
            values = [float(row["accuracy"]) for row in rows if row["split"] == split]
            plt.plot(np.arange(len(values)), values, marker=point, linestyle=linestyle, label=f"{label}/{split}")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.ylim(0, 1.0)
    plt.legend(loc="lower right")
    path = image_dir / "e13_two_layer_net.png"
    save(path)
    return [path]
