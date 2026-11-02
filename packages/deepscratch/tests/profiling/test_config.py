from dataclasses import FrozenInstanceError

import pytest
from deepscratch.profiling.config import ProfilingConfig


def test_profiling_config() -> None:
    config = ProfilingConfig()
    assert config.enabled is False
    assert config.start_step == 0
    assert config.num_steps == 10
    assert config.collect_common_metrics is True

    for kwargs in (
        {"start_step": -1},
        {"num_steps": -1},
        {"sample_memory_every_n_steps": -1},
    ):
        with pytest.raises(ValueError):
            ProfilingConfig(**kwargs)

    with pytest.raises(FrozenInstanceError):
        config.enabled = True
