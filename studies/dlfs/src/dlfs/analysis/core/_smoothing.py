"""Signal processing: Kaiser smoothing for book-replicating loss curves."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def smooth_book(values: Sequence[float]) -> np.ndarray:
    """Reproduce the book's 11-point Kaiser smoothing."""
    array = np.asarray(values, dtype=float)
    window_len = 11
    if len(array) < window_len:
        return array
    reflected = np.r_[array[window_len - 1 : 0 : -1], array, array[-1:-window_len:-1]]
    window = np.kaiser(window_len, 2)
    smoothed = np.convolve(window / window.sum(), reflected, mode="valid")
    return smoothed[5 : len(smoothed) - 5]


def smooth_histories(
    histories: Sequence[Mapping[float, float]],
) -> list[dict[float, float]]:
    output = []
    for history in histories:
        steps = sorted(history)
        values = smooth_book([history[step] for step in steps])
        output.append(
            {step: float(value) for step, value in zip(steps, values, strict=True)}
        )
    return output
