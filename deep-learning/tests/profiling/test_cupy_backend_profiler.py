import pytest

from mlprosection.core.backend import BackendConfig, make_backend
from mlprosection.profiling.backend import create_backend_profiler
from mlprosection.profiling.config import ProfilingConfig
from mlprosection.profiling.detail import DetailProfiler
from mlprosection.profiling.monitor import RuntimeMonitor

cp = pytest.importorskip("cupy")


def has_cuda_device() -> bool:
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except cp.cuda.runtime.CUDARuntimeError:
        return False


pytestmark = pytest.mark.skipif(
    not has_cuda_device(),
    reason="CUDA device is not available.",
)


def test_cupy_backend_profiler_reports_gpu_memory_and_times_section() -> None:
    backend = make_backend(BackendConfig(device="cuda", dtype="float32"))
    profiler = create_backend_profiler(backend)
    monitor = RuntimeMonitor(profiler)
    detail = DetailProfiler(ProfilingConfig(enabled=True), profiler, monitor)

    with detail.section("forward", enabled=True):
        array = backend.xp.asarray([1, 2, 3])
        _ = array + 1

    stats = profiler.memory_stats()

    assert profiler.name == "cupy"
    assert "gpu.pool_used_bytes" in stats
    assert "gpu.pool_reserved_bytes" in stats
    assert "gpu.pinned_free_blocks" in stats
    assert len(monitor.timings_ms["profile.forward"]) == 1
