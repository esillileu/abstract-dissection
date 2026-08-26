from deepscratch.core.backend import BackendConfig, make_backend
from deepscratch.profiling.backend import create_backend_profiler


def test_numpy_backend_profiler_reports_cpu_memory() -> None:
    profiler = create_backend_profiler(make_backend(BackendConfig(device="cpu")))

    profiler.synchronize()
    stats = profiler.memory_stats()

    assert profiler.name == "numpy"
    assert "cpu.rss_bytes" in stats
    assert "cpu.vms_bytes" in stats
    assert all(isinstance(value, (int, float)) for value in stats.values())
