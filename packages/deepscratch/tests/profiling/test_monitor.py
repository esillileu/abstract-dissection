import pytest
from deepscratch.profiling.monitor import RuntimeMonitor
from deepscratch.profiling.summary import training_summary


class MockBackendProfiler:
    name = "numpy"

    def __init__(self) -> None:
        self.synchronize_count = 0
        self.memory_values = [
            {"cpu.rss_bytes": 10, "cpu.vms_bytes": 20},
            {"cpu.rss_bytes": 15, "cpu.vms_bytes": 18},
        ]

    def synchronize(self) -> None:
        self.synchronize_count += 1

    def memory_stats(self) -> dict[str, int]:
        if len(self.memory_values) == 1:
            return self.memory_values[0]
        return self.memory_values.pop(0)


def test_monitor_timer_records_values_and_synchronizes() -> None:
    backend = MockBackendProfiler()
    monitor = RuntimeMonitor(backend)

    with monitor.timer("epoch", synchronize=True):
        pass

    assert len(monitor.timings_ms["epoch"]) == 1
    assert backend.synchronize_count == 2


def test_monitor_timer_records_on_exception() -> None:
    monitor = RuntimeMonitor(MockBackendProfiler())

    with pytest.raises(RuntimeError), monitor.timer("step"):
        raise RuntimeError("boom")

    assert len(monitor.timings_ms["step"]) == 1


def test_monitor_memory_snapshot_peak_and_metrics() -> None:
    monitor = RuntimeMonitor(MockBackendProfiler())

    monitor.snapshot_memory("run.start")
    monitor.snapshot_memory("run.start")
    monitor.update_memory_peaks()
    metrics = monitor.metrics()

    assert metrics["memory.run.start.cpu.rss_bytes"] == 15
    assert metrics["memory.peak_sampled.cpu.rss_bytes"] == 15


def test_monitor_rejects_non_numeric_metric() -> None:
    monitor = RuntimeMonitor(MockBackendProfiler())

    with pytest.raises(TypeError):
        monitor.set_metric("bad", "value")


def test_training_summary_can_measure_synchronized_total() -> None:
    backend = MockBackendProfiler()
    monitor = RuntimeMonitor(backend)

    with training_summary(monitor, synchronize=True):
        pass

    assert backend.synchronize_count == 2
    assert len(monitor.timings_ms["train_total"]) == 1
    assert len(monitor.timings_ms["train_synchronized"]) == 1
