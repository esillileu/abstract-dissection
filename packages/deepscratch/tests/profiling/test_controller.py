from deepscratch.profiling.config import ProfilingConfig
from deepscratch.profiling.controller import ProfilingController


def test_disabled_controller_never_profiles() -> None:
    controller = ProfilingController(ProfilingConfig(enabled=False))

    assert controller.should_profile(0) is False
    assert controller.should_profile(10) is False


def test_controller_profiles_start_inclusive_end_exclusive() -> None:
    controller = ProfilingController(
        ProfilingConfig(enabled=True, start_step=10, num_steps=3)
    )

    assert controller.should_profile(9) is False
    assert controller.should_profile(10) is True
    assert controller.should_profile(12) is True
    assert controller.should_profile(13) is False


def test_controller_num_steps_zero_profiles_nothing() -> None:
    controller = ProfilingController(ProfilingConfig(enabled=True, num_steps=0))

    assert controller.should_profile(0) is False


def test_controller_memory_sampling_interval() -> None:
    controller = ProfilingController(ProfilingConfig(sample_memory_every_n_steps=3))

    assert controller.should_sample_memory(0) is True
    assert controller.should_sample_memory(1) is False
    assert controller.should_sample_memory(3) is True


def test_controller_memory_sampling_can_be_disabled() -> None:
    controller = ProfilingController(ProfilingConfig(collect_memory_metrics=False))

    assert controller.should_sample_memory(0) is False


def test_controller_zero_memory_interval_keeps_only_boundary_snapshots() -> None:
    controller = ProfilingController(ProfilingConfig(sample_memory_every_n_steps=0))

    assert controller.should_sample_memory(0) is False
    assert controller.should_sample_memory(100) is False
