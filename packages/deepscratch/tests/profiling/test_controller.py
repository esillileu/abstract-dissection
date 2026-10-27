from deepscratch.profiling.config import ProfilingConfig
from deepscratch.profiling.controller import ProfilingController


def test_profiling_controller() -> None:
    # Disabled controller
    disabled = ProfilingController(ProfilingConfig(enabled=False))
    assert disabled.should_profile(0) is False
    assert disabled.should_profile(10) is False

    # Enabled with step boundaries
    controller = ProfilingController(
        ProfilingConfig(enabled=True, start_step=10, num_steps=3)
    )
    assert controller.should_profile(9) is False
    assert controller.should_profile(10) is True
    assert controller.should_profile(12) is True
    assert controller.should_profile(13) is False

    # Zero steps
    assert (
        ProfilingController(ProfilingConfig(enabled=True, num_steps=0)).should_profile(
            0
        )
        is False
    )

    # Memory sampling intervals
    sampled = ProfilingController(ProfilingConfig(sample_memory_every_n_steps=3))
    assert sampled.should_sample_memory(0) is True
    assert sampled.should_sample_memory(1) is False
    assert sampled.should_sample_memory(3) is True

    # Memory sampling disabled or zero interval
    assert (
        ProfilingController(
            ProfilingConfig(collect_memory_metrics=False)
        ).should_sample_memory(0)
        is False
    )
    assert (
        ProfilingController(
            ProfilingConfig(sample_memory_every_n_steps=0)
        ).should_sample_memory(0)
        is False
    )
