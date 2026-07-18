from __future__ import annotations

from dataclasses import dataclass


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
    sample_memory_every_n_steps: int = 1

    def __post_init__(self) -> None:
        if self.start_step < 0:
            raise ValueError("start_step must be >= 0")
        if self.num_steps < 0:
            raise ValueError("num_steps must be >= 0")
        if self.sample_memory_every_n_steps < 1:
            raise ValueError("sample_memory_every_n_steps must be >= 1")
