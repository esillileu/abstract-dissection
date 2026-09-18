from __future__ import annotations

import gc
from dataclasses import asdict
from math import sqrt

import numpy as np
from deepscratch.profiling import BenchmarkRunner

from .constants import (
    _T_975,
    EMBEDDING_SIZE,
)
from .workload import ScalingWorkload


def _measure_condition(
    condition: str,
    *,
    vocab_size: int,
    contexts: np.ndarray,
    targets: np.ndarray,
    backend,
    batch_size: int,
    warmup_updates: int,
    measured_updates: int,
    repetitions: int,
) -> dict[str, object]:
    model_name = "CBOW" if "-cbow-" in condition else "SkipGram"
    objective_name = (
        "FusedNegativeSampling"
        if condition.endswith("-fused-ns")
        else "NegativeSampling"
        if condition.endswith("-ns")
        else "FullSoftmax"
    )
    row: dict[str, object] = {
        "condition": condition,
        "implementation": "implemented",
        "model": model_name,
        "objective": objective_name,
        "vocab_size": vocab_size,
        "embedding_size": EMBEDDING_SIZE,
        "batch_size": batch_size,
        "warmup_updates": warmup_updates,
        "measured_updates": measured_updates,
        "repetitions": repetitions,
        "dense_parameter_optimizer_bytes": 8
        * vocab_size
        * EMBEDDING_SIZE
        * np.dtype(np.float32).itemsize,
    }
    workload = None
    next_index = 0
    try:
        workload = ScalingWorkload(
            condition,
            vocab_size=vocab_size,
            contexts=contexts,
            targets=targets,
            backend=backend,
        )

        def update_once() -> None:
            nonlocal next_index
            workload.update(next_index, batch_size)
            next_index += 1

        result = BenchmarkRunner(backend).measure_update_protocol(
            f"vocabulary_size_scaling.v{vocab_size}.{condition}",
            update_once,
            warmup_iterations=warmup_updates,
            measured_iterations=measured_updates,
            repetitions=repetitions,
        )
        interval = _mean_confidence_interval_95(result.timing)
        row.update(
            {
                "status": "ok",
                "update_ms": result.timing.mean_ms,
                "standard_error_ms": interval["standard_error_ms"],
                "ci95_lower_ms": interval["lower_ms"],
                "ci95_upper_ms": interval["upper_ms"],
                "ci95_half_width_ms": interval["half_width_ms"],
                "cold_ms": result.cold_ms,
                "warmup_total_ms": result.warmup_total_ms,
                "warmup_mean_ms": result.warmup_mean_ms,
                "steady_event_timing": asdict(result.event_timing),
                "timing": asdict(result.timing),
                "error": None,
            }
        )
    except Exception as exc:
        if not _is_out_of_memory(exc):
            raise
        row.update(
            {
                "status": "out_of_memory",
                "update_ms": None,
                "standard_error_ms": None,
                "ci95_lower_ms": None,
                "ci95_upper_ms": None,
                "ci95_half_width_ms": None,
                "cold_ms": None,
                "warmup_total_ms": None,
                "warmup_mean_ms": None,
                "steady_event_timing": None,
                "timing": None,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
    finally:
        workload = None
        gc.collect()
        _release_backend_memory(backend)
    return row


def measure_scaling_condition(*args, **kwargs) -> dict[str, object]:
    """Public compatibility API for one vocabulary-size scaling point."""
    return _measure_condition(*args, **kwargs)


def _mean_confidence_interval_95(timing) -> dict[str, float | None]:
    if timing.count < 2:
        return {
            "standard_error_ms": None,
            "lower_ms": None,
            "upper_ms": None,
            "half_width_ms": None,
        }
    standard_error = timing.stdev_ms / sqrt(timing.count)
    degrees_of_freedom = timing.count - 1
    critical_value = (
        _T_975[degrees_of_freedom] if degrees_of_freedom < len(_T_975) else 1.96
    )
    half_width = critical_value * standard_error
    return {
        "standard_error_ms": standard_error,
        "lower_ms": max(0.0, timing.mean_ms - half_width),
        "upper_ms": timing.mean_ms + half_width,
        "half_width_ms": half_width,
    }


def _is_out_of_memory(exc: Exception) -> bool:
    return isinstance(exc, MemoryError) or type(exc).__name__ == "OutOfMemoryError"


def _release_backend_memory(backend) -> None:
    if not backend.is_gpu:
        return
    backend.synchronize()
    backend.xp.get_default_memory_pool().free_all_blocks()
    backend.xp.get_default_pinned_memory_pool().free_all_blocks()
