"""Tiling and mosaic layout utilities for 2D filter visualization."""

from __future__ import annotations

import math

import numpy as np


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
