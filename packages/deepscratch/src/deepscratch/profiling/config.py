from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields


@dataclass(frozen=True)
class ProfilingConfig:
    """Runtime metric and optional detailed profiling settings."""

    enabled: bool = False
    start_step: int = 0
    num_steps: int = 10
    profile_python: bool = False
    profile_memory: bool = False
    profile_gpu_ranges: bool = False
    collect_common_metrics: bool = True
    collect_epoch_metrics: bool = True
    collect_memory_metrics: bool = True
    collect_model_metrics: bool = True
    # Synchronize only at whole-training boundaries for completed GPU time.
    synchronize_train: bool = False
    # Zero disables per-update peak sampling while retaining run/epoch snapshots.
    sample_memory_every_n_steps: int = 1

    def __post_init__(self) -> None:
        if self.start_step < 0:
            raise ValueError("start_step must be >= 0")
        if self.num_steps < 0:
            raise ValueError("num_steps must be >= 0")
        if self.sample_memory_every_n_steps < 0:
            raise ValueError("sample_memory_every_n_steps must be >= 0")


def profiling_config_from_mapping(values: Mapping[str, object]) -> ProfilingConfig:
    """Build a validated profiling config from a YAML profiling section."""
    allowed = {field.name for field in fields(ProfilingConfig)}
    return ProfilingConfig(
        **{key: value for key, value in values.items() if key in allowed}
    )
