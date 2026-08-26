"""Generic synchronized measurement engine for atomic profile workloads."""

from __future__ import annotations

from dataclasses import asdict
from math import sqrt

from deepscratch.profiling import BenchmarkRunner

from .contracts import MeasurementProtocol, ProfilePoint, ProfileWorkload


def measure_update_workload(
    condition_id: str,
    workload: ProfileWorkload,
    *,
    axes: dict[str, int | float | str],
    protocol: MeasurementProtocol,
) -> ProfilePoint:
    try:
        result = BenchmarkRunner(workload.backend).measure_update_protocol(
            condition_id,
            workload.update,
            warmup_iterations=protocol.warmup_iterations,
            measured_iterations=protocol.measured_iterations,
            repetitions=protocol.repetitions,
        )
        timing = (
            result.event_timing if protocol.timing_source == "event" else result.timing
        )
        interval = _confidence_interval_95(
            timing.mean_ms, timing.stdev_ms, timing.count
        )
        return ProfilePoint(
            condition_id=condition_id,
            axes=axes,
            status="ok",
            metrics={
                "update_ms": timing.mean_ms,
                "update_stdev_ms": timing.stdev_ms,
                "cold_ms": result.cold_ms,
                "p50_ms": timing.p50_ms,
                "p95_ms": timing.p95_ms,
                "ci95_lower_ms": interval[0],
                "ci95_upper_ms": interval[1],
                "window_update_ms": result.timing.mean_ms,
                "event_update_ms": result.event_timing.mean_ms,
            },
            timings={
                "window": asdict(result.timing),
                "event": asdict(result.event_timing),
            },
        )
    except Exception as error:
        if not _is_out_of_memory(error):
            raise
        return ProfilePoint(
            condition_id=condition_id,
            axes=axes,
            status="out_of_memory",
            metrics={},
            error=f"{type(error).__name__}: {error}",
        )
    finally:
        workload.release()


def measure_workload_sections(
    workload: ProfileWorkload,
    *,
    warmup_iterations: int,
    measured_iterations: int,
) -> dict[str, dict[str, object]]:
    if warmup_iterations < 0 or measured_iterations < 1:
        raise ValueError("section warmup must be non-negative and measurement positive")
    runner = BenchmarkRunner(workload.backend)
    try:
        return {
            name: runner.measure_iterations(
                name,
                section.operation,
                prepare=section.prepare,
                warmup_iterations=warmup_iterations,
                measured_iterations=measured_iterations,
            ).to_dict()
            for name, section in workload.sections().items()
        }
    finally:
        workload.release()


def _is_out_of_memory(error: Exception) -> bool:
    return isinstance(error, MemoryError) or type(error).__name__ == "OutOfMemoryError"


def _confidence_interval_95(
    mean_ms: float, stdev_ms: float, count: int
) -> tuple[float | None, float | None]:
    if count < 2:
        return None, None
    critical = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
    }.get(count - 1, 1.96)
    half_width = critical * stdev_ms / sqrt(count)
    return max(0.0, mean_ms - half_width), mean_ms + half_width


__all__ = ["measure_update_workload", "measure_workload_sections"]
