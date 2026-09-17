"""Training time extrapolation from benchmark timing statistics."""

from __future__ import annotations

from .results import TimingStats, TrainingTimeEstimate


def estimate_training_time(
    update_timing: TimingStats,
    *,
    dataset_samples: int,
    batch_size: int,
    epochs: int,
    drop_last: bool = True,
    cold_update_ms: float | None = None,
) -> TrainingTimeEstimate:
    """Extrapolate update timing, optionally charging one cold update once."""
    if min(dataset_samples, batch_size, epochs) < 1:
        raise ValueError("dataset_samples, batch_size, and epochs must be positive")
    updates_per_epoch = (
        dataset_samples // batch_size
        if drop_last
        else (dataset_samples + batch_size - 1) // batch_size
    )
    total_updates = updates_per_epoch * epochs
    if cold_update_ms is None:
        steady_updates = total_updates
        mean_total_ms = update_timing.mean_ms * steady_updates
    else:
        if cold_update_ms < 0:
            raise ValueError("cold_update_ms must be non-negative")
        steady_updates = max(total_updates - 1, 0)
        mean_total_ms = cold_update_ms + update_timing.mean_ms * steady_updates
    repeat_stdev_total_ms = update_timing.stdev_ms * steady_updates
    return TrainingTimeEstimate(
        updates_per_epoch=updates_per_epoch,
        epochs=epochs,
        mean_seconds_per_epoch=mean_total_ms / epochs / 1_000,
        repeat_stdev_seconds_per_epoch=(repeat_stdev_total_ms / epochs / 1_000),
        mean_seconds_total=mean_total_ms / 1_000,
        repeat_stdev_seconds_total=repeat_stdev_total_ms / 1_000,
    )


__all__ = ["estimate_training_time"]
