from __future__ import annotations

from contextlib import nullcontext

from mlprosection.profiling import (
    BenchmarkRunner,
    TimingStats,
    estimate_training_time,
)


class _CpuBackend:
    is_gpu = False

    def synchronize(self) -> None:
        pass

    def range(self, _name: str):
        return nullcontext()


def test_update_protocol_separates_cold_events_and_windows() -> None:
    calls = 0

    def operation() -> None:
        nonlocal calls
        calls += 1

    result = BenchmarkRunner(_CpuBackend()).measure_update_protocol(
        "update",
        operation,
        warmup_iterations=2,
        measured_iterations=3,
        repetitions=4,
    )

    assert calls == 1 + 2 + 3 + 3 * 4
    assert result.event_timing.count == 3
    assert result.timing.count == 4
    assert result.cold_ms >= 0


def test_training_estimate_charges_cold_update_only_once() -> None:
    timing = TimingStats(
        count=5,
        mean_ms=2.0,
        stdev_ms=0.5,
        min_ms=2.0,
        max_ms=2.0,
        p50_ms=2.0,
        p95_ms=2.0,
    )

    estimate = estimate_training_time(
        timing,
        dataset_samples=10,
        batch_size=2,
        epochs=2,
        cold_update_ms=11.0,
    )

    assert estimate.updates_per_epoch == 5
    assert estimate.mean_seconds_total == 0.029
    assert estimate.mean_seconds_per_epoch == 0.0145
    assert estimate.repeat_stdev_seconds_total == 0.0045
    assert estimate.repeat_stdev_seconds_per_epoch == 0.00225
