"""Render the five persisted attention maps."""

from pathlib import Path

from repro_core.plotting.theme import INK

from .common import load_csv, load_npz, np, plt, save, trial

TRIAL_IDS = ("dlfs2.ch08.attention-alignment",)


def render(root: Path, image_dir: Path) -> list[Path]:
    directory = trial(root, "e08", TRIAL_IDS[0])
    arrays = load_npz(directory / "attention.npz")
    labels = load_csv(directory / "labels.csv")
    outputs = []
    for index, row in enumerate(labels):
        attention = arrays[f"attention_{index}"]
        _figure, axis = plt.subplots()
        axis.pcolor(attention, cmap=plt.cm.Greys_r, vmin=0.0, vmax=1.0)
        axis.patch.set_facecolor(INK)
        axis.set_yticks(np.arange(attention.shape[0]) + 0.5, minor=False)
        axis.set_xticks(np.arange(attention.shape[1]) + 0.5, minor=False)
        axis.invert_yaxis()
        axis.set_xticklabels(list(row["row_labels"]), minor=False)
        axis.set_yticklabels(list(row["column_labels"]), minor=False)
        path = image_dir / f"e08_attention_{index + 1}.png"
        save(path)
        outputs.append(path)
    return outputs
