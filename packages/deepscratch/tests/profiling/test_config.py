from dataclasses import FrozenInstanceError

import pytest
from deepscratch.profiling.config import ProfilingConfig


def test_profiling_config_defaults() -> None:
    config = ProfilingConfig()

    assert config.enabled is False
    assert config.start_step == 0
    assert config.num_steps == 10
    assert config.collect_common_metrics is True


def test_profiling_config_rejects_negative_start_step() -> None:
    with pytest.raises(ValueError):
        ProfilingConfig(start_step=-1)


def test_profiling_config_rejects_negative_num_steps() -> None:
    with pytest.raises(ValueError):
        ProfilingConfig(num_steps=-1)


def test_profiling_config_rejects_negative_memory_interval() -> None:
    with pytest.raises(ValueError):
        ProfilingConfig(sample_memory_every_n_steps=-1)


def test_profiling_config_is_frozen() -> None:
    config = ProfilingConfig()

    with pytest.raises(FrozenInstanceError):
        config.enabled = True
