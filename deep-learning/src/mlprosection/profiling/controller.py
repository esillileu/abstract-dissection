from __future__ import annotations

from .config import ProfilingConfig


class ProfilingController:
    """Decides when step profiling and memory sampling should run."""

    def __init__(self, config: ProfilingConfig) -> None:
        self.config = config

    def should_profile(self, global_step: int) -> bool:
        if not self.config.enabled or self.config.num_steps == 0:
            return False

        start = self.config.start_step
        end = start + self.config.num_steps
        return start <= global_step < end

    def should_sample_memory(self, global_step: int) -> bool:
        if not self.config.collect_memory_metrics:
            return False

        return global_step % self.config.sample_memory_every_n_steps == 0
