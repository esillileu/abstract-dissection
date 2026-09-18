from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConditionResult:
    condition: str
    implementation: str
    model: str
    objective: str
    batch_size: int
    dataset_samples: int
    updates_per_epoch: int
    estimated_epochs: int
    warmup_updates: int
    measured_updates: int
    repetitions: int
    cold_ms_per_update: float
    warmup_total_ms: float
    warmup_mean_ms: float
    steady_event_mean_ms_per_update: float
    steady_event_stdev_ms_per_update: float
    steady_event_min_ms_per_update: float
    steady_event_max_ms_per_update: float
    steady_event_p50_ms_per_update: float
    steady_event_p95_ms_per_update: float
    mean_ms_per_update: float
    stdev_ms_per_update: float
    min_ms_per_update: float
    max_ms_per_update: float
    samples_per_second: float
    estimated_first_epoch_seconds: float
    estimated_seconds_per_epoch: float
    estimated_repeat_stdev_seconds_per_epoch: float
    estimated_seconds_total: float
    estimated_repeat_stdev_seconds_total: float
    phase_ms_per_update: dict[str, float]
    phase_stats: dict[str, dict[str, float | int]]
    phase_share: dict[str, float]
