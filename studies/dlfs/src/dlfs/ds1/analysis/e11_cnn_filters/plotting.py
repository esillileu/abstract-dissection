"""Figure rendering and CSV writing for CNN filter visualization."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from repro_core.plotting.theme import SURFACE

from .tiling import _filter_mosaic

FILTER_COLOR_LIMIT = 0.9
SUMMARY_FIELDS = (
    "group",
    "condition",
    "seed",
    "run_id",
    "checkpoint_format",
    "checkpoint_epoch",
    "checkpoint_update",
    "parameter",
    "shape",
    "weight_min",
    "weight_max",
    "weight_mean",
    "weight_std",
    "image",
)


def _get_save_figure():
    mod = sys.modules.get("dlfs.ds1.analysis.e11_cnn_filters")
    if mod is not None and hasattr(mod, "save_figure"):
        return mod.save_figure
    from repro_core.analysis.core import save_figure

    return save_figure


def _shared_weight_limit(weight_sets: list[np.ndarray]) -> float:
    limit = max(
        (float(np.max(np.abs(weights))) for weights in weight_sets if weights.size),
        default=1.0,
    )
    return limit if limit > 0.0 else 1.0


def _render_panel(
    panel: tuple[str, str, np.ndarray],
    *,
    output: Path,
    limit: float,
) -> None:
    del limit
    _group, _condition, weights = panel
    figure = plt.figure(figsize=(6.6, 6))
    grid = figure.add_gridspec(1, 2, width_ratios=[1.0, 0.035], wspace=0.08)
    axis = figure.add_subplot(grid[0, 0])
    color_axis = figure.add_subplot(grid[0, 1])
    color_map = plt.colormaps["gray_r"].copy()
    color_map.set_bad(SURFACE)
    image = axis.imshow(
        _filter_mosaic(weights),
        cmap=color_map,
        interpolation="nearest",
        vmin=-FILTER_COLOR_LIMIT,
        vmax=FILTER_COLOR_LIMIT,
    )
    axis.set_xticks(())
    axis.set_yticks(())
    figure.colorbar(
        image,
        cax=color_axis,
        label="weight (shared scale)",
    )
    figure._analysis_skip_tight_layout = True
    _get_save_figure()(figure, output)
    plt.close(figure)


def _panel_output(output: Path, group: str, condition: str) -> Path:
    suffix = f"_{group.lower()}_{condition.lower()}"
    return output.with_name(f"{output.stem}{suffix}{output.suffix}")


def _write_summary(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path
