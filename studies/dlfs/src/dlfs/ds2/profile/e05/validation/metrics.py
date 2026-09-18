"""Metric evaluation and numerical tolerance contracts for e05 validation."""

from __future__ import annotations

import numpy as np

FORWARD_CEILING = 1e-4
GRADIENT_CEILING = 1e-3
TOLERANCE_GRID = (1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3)


def _host(backend, value) -> np.ndarray:
    return np.asarray(backend.to_numpy(value))


def error_metrics(backend, reference, actual) -> dict[str, float]:
    left, right = _host(backend, reference), _host(backend, actual)
    absolute = np.abs(left.astype(np.float64) - right.astype(np.float64))
    scale = np.maximum(np.maximum(np.abs(left), np.abs(right)), 1e-6)
    return {
        "max_absolute": float(absolute.max(initial=0.0)),
        "max_relative": float((absolute / scale).max(initial=0.0)),
        "required_atol_rtol": float((absolute / (1.0 + scale)).max(initial=0.0)),
    }


def _within(metrics: dict[str, float], ceiling: float) -> bool:
    # This is the standard combined atol/rtol test expressed using the two
    # recorded maxima. The raw maxima remain in the artifact for inspection.
    return metrics["required_atol_rtol"] <= ceiling


def _selected_tolerance(maximum: float, ceiling: float) -> float | None:
    target = 2 * maximum
    return next(
        (value for value in TOLERANCE_GRID if value >= target and value <= ceiling),
        None,
    )
