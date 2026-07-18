from mlprosection.profiling.config import ProfilingConfig
from mlprosection.profiling.detail import DetailProfiler
from mlprosection.profiling.monitor import RuntimeMonitor


class MockBackendProfiler:
    name = "numpy"

    def synchronize(self) -> None:
        pass

    def memory_stats(self) -> dict[str, int]:
        return {"cpu.rss_bytes": 1, "cpu.vms_bytes": 2}


def test_detail_profiler_section_noop_when_disabled() -> None:
    monitor = RuntimeMonitor(MockBackendProfiler())
    profiler = DetailProfiler(ProfilingConfig(), MockBackendProfiler(), monitor)

    with profiler.section("forward", enabled=False):
        pass

    assert "profile.forward" not in monitor.timings_ms


def test_detail_profiler_section_records_timer_when_enabled() -> None:
    monitor = RuntimeMonitor(MockBackendProfiler())
    backend = MockBackendProfiler()
    profiler = DetailProfiler(ProfilingConfig(enabled=True), backend, monitor)

    with profiler.section("forward", enabled=True):
        pass

    assert len(monitor.timings_ms["profile.forward"]) == 1


def test_detail_profiler_python_artifacts(tmp_path) -> None:
    monitor = RuntimeMonitor(MockBackendProfiler())
    profiler = DetailProfiler(
        ProfilingConfig(profile_python=True),
        MockBackendProfiler(),
        monitor,
    )

    profiler.start_run()
    profiler.stop_run()
    artifacts = profiler.dump_artifacts(tmp_path)

    assert {artifact.name for artifact in artifacts} == {
        "python_profile.prof",
        "python_profile.txt",
    }


def test_detail_profiler_tracemalloc_records_metrics() -> None:
    monitor = RuntimeMonitor(MockBackendProfiler())
    profiler = DetailProfiler(
        ProfilingConfig(profile_memory=True),
        MockBackendProfiler(),
        monitor,
    )

    profiler.start_run()
    profiler.stop_run()

    assert "memory.python_traced.current_bytes" in monitor.scalar_metrics
    assert "memory.python_traced.peak_bytes" in monitor.scalar_metrics
