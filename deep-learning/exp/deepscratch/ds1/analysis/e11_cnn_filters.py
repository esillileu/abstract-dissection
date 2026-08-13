"""DS1 E11: compare seed-index 0 SimpleCNN filters on one shared scale."""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from exp.analyze import artifact_file, mark_empty, save_figure
from exp.plot_theme import SURFACE

from .common import runs


RUN_GROUPS = (
    ("GT06", "CNN-SIMPLE-BOOK"),
    ("GT08", "CNN-SIMPLE-SPATIAL"),
    ("GT08", "CNN-SIMPLE-SPATIAL-PERMUTED"),
)
# Index 0 in exp/deepscratch/ds1/config/implemented/seeds.yaml (research_v1).
VISUALIZATION_SEED = "1"
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


def _natural_key(value: str) -> tuple[object, ...]:
    return tuple(
        int(part) if part.isdigit() else part for part in re.split(r"(\d+)", value)
    )


def _conv_weights(checkpoint: Path) -> list[tuple[str, np.ndarray]]:
    """Return convolution kernels in model order from a v1 or v2 checkpoint."""
    with np.load(checkpoint, allow_pickle=False) as arrays:
        weights = [
            (name, np.asarray(arrays[name]))
            for name in arrays.files
            if name.endswith(".W") and arrays[name].ndim == 4
        ]
    return sorted(weights, key=lambda item: _natural_key(item[0]))


def _square_grid(count: int, *, maximum_columns: int = 8) -> tuple[int, int]:
    columns = min(maximum_columns, max(1, math.ceil(math.sqrt(count))))
    return math.ceil(count / columns), columns


def _tiled_array(images: np.ndarray, *, maximum_columns: int = 8) -> np.ndarray:
    """Tile N grayscale images with NaN gutters."""
    count, height, width = images.shape
    rows, columns = _square_grid(count, maximum_columns=maximum_columns)
    canvas = np.full(
        (rows * height + rows - 1, columns * width + columns - 1),
        np.nan,
        dtype=float,
    )
    for index, image in enumerate(images):
        row, column = divmod(index, columns)
        top = row * (height + 1)
        left = column * (width + 1)
        canvas[top : top + height, left : left + width] = image
    return canvas


def _filter_mosaic(weights: np.ndarray) -> np.ndarray:
    """Make one tile per output filter, retaining every input-channel kernel."""
    if weights.ndim != 4:
        raise ValueError(
            f"convolution weights must be four-dimensional, got {weights.shape}"
        )
    filter_tiles = np.asarray(
        [_tiled_array(output_filter) for output_filter in weights],
        dtype=float,
    )
    return _tiled_array(filter_tiles)


def _checkpoint_weights_path(client, run):
    manifest_path = artifact_file(
        client,
        run,
        "checkpoints/checkpoint_manifest.json",
    )
    if manifest_path is None:
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        final = manifest.get("final")
        if not isinstance(final, dict) or not final.get("path"):
            return None
        final_path = Path(str(final["path"]))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None

    candidates = []
    if final_path.is_dir():
        candidates.append(final_path / "model_parameters.npz")
    else:
        candidates.append(final_path)
    if run.local_artifact_root is not None:
        candidates.extend(
            (
                run.local_artifact_root.parent.parent
                / "checkpoints"
                / run.local_artifact_root.name
                / "final.npz",
                run.local_artifact_root
                / "checkpoints"
                / final_path.name
                / "model_parameters.npz",
            )
        )
    for candidate in candidates:
        if candidate.is_file():
            return candidate, manifest

    if final_path.suffix == ".npz":
        remote_path = "checkpoints/final.npz"
    else:
        remote_path = f"checkpoints/{final_path.name}/model_parameters.npz"
    downloaded = artifact_file(client, run, remote_path)
    if downloaded is None:
        return None
    return downloaded, manifest


def _shared_weight_limit(weight_sets: list[np.ndarray]) -> float:
    limit = max(
        (float(np.max(np.abs(weights))) for weights in weight_sets if weights.size),
        default=1.0,
    )
    return limit if limit > 0.0 else 1.0


def _render_comparison(
    panels: list[tuple[str, str, np.ndarray]],
    *,
    output: Path,
) -> None:
    figure = plt.figure(figsize=(6 * len(panels) + 0.6, 6))
    grid = figure.add_gridspec(
        1,
        len(panels) + 1,
        width_ratios=[1.0] * len(panels) + [0.035],
        wspace=0.08,
    )
    axes = [figure.add_subplot(grid[0, index]) for index in range(len(panels))]
    color_axis = figure.add_subplot(grid[0, -1])
    limit = _shared_weight_limit([weights for _group, _condition, weights in panels])
    color_map = plt.colormaps["gray_r"].copy()
    color_map.set_bad(SURFACE)
    images = []
    for axis, (group, condition, weights) in zip(axes, panels, strict=True):
        images.append(
            axis.imshow(
                _filter_mosaic(weights),
                cmap=color_map,
                interpolation="nearest",
                vmin=-limit,
                vmax=limit,
            )
        )
        axis.set_title(f"{group} | {condition}\n{tuple(weights.shape)}")
        axis.set_xticks(())
        axis.set_yticks(())
    figure.colorbar(
        images[0],
        cax=color_axis,
        label="weight (shared scale)",
    )
    figure.suptitle(
        f"SimpleCNN first-layer filters | seed index 0 (master {VISUALIZATION_SEED})"
    )
    figure._analysis_skip_tight_layout = True
    save_figure(figure, output)
    plt.close(figure)


def _write_summary(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _visualization_runs(run_refs):
    return [run for run in run_refs if run.seed == VISUALIZATION_SEED]


def _collect(
    client, *, image: str
) -> tuple[
    list[tuple[str, str, np.ndarray]],
    list[dict[str, object]],
]:
    panels: list[tuple[str, str, np.ndarray]] = []
    summary_rows: list[dict[str, object]] = []
    for group, condition in RUN_GROUPS:
        condition_runs = runs(client, group, [condition])[condition]
        for run in _visualization_runs(condition_runs):
            checkpoint = _checkpoint_weights_path(client, run)
            if checkpoint is None:
                continue
            checkpoint_path, manifest = checkpoint
            try:
                layers = _conv_weights(checkpoint_path)
            except (OSError, ValueError):
                continue
            if not layers:
                continue
            final = manifest["final"]
            checkpoint_format = str(manifest.get("format", "unknown"))
            parameter, weights = layers[0]
            panels.append((group, condition, weights))
            summary_rows.append(
                {
                    "group": group,
                    "condition": condition,
                    "seed": run.seed,
                    "run_id": run.run_id,
                    "checkpoint_format": checkpoint_format,
                    "checkpoint_epoch": final.get("epoch", ""),
                    "checkpoint_update": final.get("update", ""),
                    "parameter": parameter,
                    "shape": "x".join(str(size) for size in weights.shape),
                    "weight_min": float(weights.min()),
                    "weight_max": float(weights.max()),
                    "weight_mean": float(weights.mean()),
                    "weight_std": float(weights.std()),
                    "image": image,
                }
            )
    return panels, summary_rows


def render(client, error_style, output):
    del error_style
    panels, summary_rows = _collect(client, image=output.as_posix())
    if panels:
        _render_comparison(panels, output=output)
    else:
        figure, axis = plt.subplots(figsize=(8, 4))
        mark_empty(axis, "No completed seed-index 0 runs with final checkpoints")
        save_figure(figure, output)
        plt.close(figure)
    summary = output.with_suffix(".csv")
    _write_summary(summary, summary_rows)
    return [output, summary]


def render_summary(client, error_style, output):
    """Write final-checkpoint filter statistics without rendering an image."""
    del error_style
    _panels, summary_rows = _collect(client, image="")
    summary = output.with_suffix(".csv")
    _write_summary(summary, summary_rows)
    for row in summary_rows:
        print(
            f"[{row['condition']}] {row['parameter']} {row['shape']}: "
            f"mean={float(row['weight_mean']):.4f}, "
            f"std={float(row['weight_std']):.4f}, "
            f"min={float(row['weight_min']):.4f}, "
            f"max={float(row['weight_max']):.4f}"
        )
    return [summary]
