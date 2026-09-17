"""Numerical error metrics and tolerance ceilings for e02 fused validation."""

from __future__ import annotations

import numpy as np
from deepscratch.optim.SGD import Adam

LOSS_CEILING = 1e-5
GRADIENT_CEILING = 1e-4
PARAMETER_CEILING = 1e-4


def _error_metrics(backend, expected, actual) -> dict[str, float]:
    left = np.asarray(backend.to_numpy(expected), dtype=np.float64)
    right = np.asarray(backend.to_numpy(actual), dtype=np.float64)
    absolute = np.abs(left - right)
    scale = np.maximum(np.maximum(np.abs(left), np.abs(right)), 1e-6)
    return {
        "max_absolute": float(absolute.max(initial=0.0)),
        "max_relative": float((absolute / scale).max(initial=0.0)),
        "required_atol_rtol": float((absolute / (1.0 + scale)).max(initial=0.0)),
    }


def _parameter_errors(backend, source, target) -> dict[str, dict[str, float]]:
    other = dict(target.named_parameters())
    return {
        name: _error_metrics(backend, parameter.data, other[name].data)
        for name, parameter in source.named_parameters()
    }


def _optimizer_errors(
    backend, source: Adam, target: Adam
) -> dict[str, dict[str, float]]:
    return {
        f"m.{name}": _error_metrics(backend, source.m[name], target.m[name])
        for name in source.m
    } | {
        f"v.{name}": _error_metrics(backend, source.v[name], target.v[name])
        for name in source.v
    }


def _passed(metrics: dict[str, dict[str, float]], ceiling: float) -> bool:
    return all(value["required_atol_rtol"] <= ceiling for value in metrics.values())


def _max_error(rows: list[dict[str, object]], group: str, field: str) -> float:
    return max(
        float(metric[field])
        for row in rows
        for metric in (row[group].values() if group != "loss_error" else (row[group],))
    )
