"""Render initial and trained ch07 convolution filters from saved host arrays."""

from pathlib import Path

from .common import load_npz, np, plt, save, trial


def _filter_show(filters, path: Path) -> None:
    count = filters.shape[0]
    rows = int(np.ceil(count / 8))
    figure = plt.figure()
    figure.subplots_adjust(
        left=0, right=1, bottom=0, top=0.9, hspace=0.05, wspace=0.05
    )
    figure.suptitle("Original SimpleCNN first-layer filters")
    for index in range(count):
        axis = figure.add_subplot(rows, 8, index + 1, xticks=[], yticks=[])
        axis.imshow(
            filters[index, 0],
            cmap=plt.cm.gray_r,
            interpolation="nearest",
        )
    save(path)


def render(root: Path, image_dir: Path) -> list[Path]:
    arrays = load_npz(
        trial(root, "e06", "dlfs1.ch07.simple-convnet") / "checkpoint.npz"
    )
    initial = image_dir / "e11_filters_initial.png"
    trained = image_dir / "e11_filters_trained.png"
    _filter_show(arrays["initial_W1"], initial)
    _filter_show(arrays["param__W1"], trained)
    return [initial, trained]
