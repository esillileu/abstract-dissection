from __future__ import annotations

import inspect
import types
from collections.abc import Iterable
from typing import Any


def summarize_values(values: list[float]) -> dict[str, float]:
    """Return summary statistics for a list of timing values."""

    if not values:
        return {}

    sorted_values = sorted(values)
    count = len(sorted_values)
    mean = sum(sorted_values) / count
    variance = sum((value - mean) ** 2 for value in sorted_values) / count

    return {
        "count": count,
        "mean": mean,
        "std": variance**0.5,
        "min": sorted_values[0],
        "max": sorted_values[-1],
        "p50": _percentile(sorted_values, 50),
        "p95": _percentile(sorted_values, 95),
    }


def count_parameter_elements(model) -> int:
    """Count unique model parameter elements via named_parameters()."""

    total = 0
    for _, parameter in _unique_named_parameters(model):
        total += int(parameter.data.size)
    return total


def count_parameter_bytes(model) -> int:
    """Count unique model parameter bytes via named_parameters()."""

    total = 0
    for _, parameter in _unique_named_parameters(model):
        total += int(parameter.data.nbytes)
    return total


def count_gradient_bytes(model) -> int:
    """Count bytes held by unique parameter gradients."""

    total = 0
    for _, parameter in _unique_named_parameters(model):
        grad = getattr(parameter, "grad", None)
        if grad is None:
            continue
        total += _array_nbytes(grad)
    return total


def count_optimizer_state_bytes(optimizer) -> int:
    """Count array-like optimizer state memory, excluding parameters."""

    excluded_arrays: set[int] = set()
    for _, parameter in getattr(optimizer, "params", ()):
        excluded_arrays.add(id(parameter.data))
        grad = getattr(parameter, "grad", None)
        if grad is not None:
            excluded_arrays.add(id(grad))
            grad_data = getattr(grad, "data", None)
            if grad_data is not None and grad_data is not grad:
                excluded_arrays.add(id(grad_data))

    return _count_state_bytes(
        optimizer,
        excluded_arrays=excluded_arrays,
        seen_objects=set(),
        seen_arrays=set(),
    )


def _percentile(sorted_values: list[float], percentile: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]

    position = (len(sorted_values) - 1) * percentile / 100
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def _unique_named_parameters(model) -> Iterable[tuple[str, Any]]:
    seen: set[int] = set()
    for name, parameter in model.named_parameters():
        parameter_id = id(parameter)
        if parameter_id in seen:
            continue
        seen.add(parameter_id)
        yield name, parameter


def _array_nbytes(value: Any) -> int:
    array = getattr(value, "data", value)
    nbytes = getattr(array, "nbytes", 0)
    return int(nbytes)


def _is_array_like(value: Any) -> bool:
    return hasattr(value, "shape") and hasattr(value, "nbytes")


def _count_state_bytes(
    value: Any,
    *,
    excluded_arrays: set[int],
    seen_objects: set[int],
    seen_arrays: set[int],
) -> int:
    object_id = id(value)
    if object_id in seen_objects:
        return 0
    seen_objects.add(object_id)

    if value is None or isinstance(value, (str, bytes, int, float, bool)):
        return 0
    if callable(value) or inspect.ismodule(value) or inspect.isclass(value):
        return 0
    if isinstance(value, types.MappingProxyType):
        return 0

    data = getattr(value, "data", None)
    if data is not None and data is not value and _is_array_like(data):
        array_id = id(data)
        if array_id not in excluded_arrays and array_id not in seen_arrays:
            seen_arrays.add(array_id)
            return int(data.nbytes)

    if _is_array_like(value):
        array_id = id(value)
        if array_id in excluded_arrays or array_id in seen_arrays:
            return 0
        seen_arrays.add(array_id)
        return int(value.nbytes)

    if isinstance(value, dict):
        return sum(
            _count_state_bytes(
                item,
                excluded_arrays=excluded_arrays,
                seen_objects=seen_objects,
                seen_arrays=seen_arrays,
            )
            for key, item in value.items()
            if key not in {"params", "named_params"}
        )

    if isinstance(value, (list, tuple, set)):
        return sum(
            _count_state_bytes(
                item,
                excluded_arrays=excluded_arrays,
                seen_objects=seen_objects,
                seen_arrays=seen_arrays,
            )
            for item in value
        )

    if hasattr(value, "__dict__"):
        return _count_state_bytes(
            vars(value),
            excluded_arrays=excluded_arrays,
            seen_objects=seen_objects,
            seen_arrays=seen_arrays,
        )

    return 0
