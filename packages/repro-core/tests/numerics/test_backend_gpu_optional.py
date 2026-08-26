import pytest

from repro_core.numerics import BackendConfig, make_backend

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


def test_make_gpu_backend_basic_fields() -> None:
    backend = make_backend(BackendConfig(device="cuda", dtype="float32"))

    assert backend.name == "cupy"
    assert backend.device == "cuda:0"
    assert backend.is_gpu
    assert not backend.is_cpu
    assert backend.dtype_name == "float32"


def test_gpu_backend_asfloat_and_to_numpy() -> None:
    backend = make_backend(BackendConfig(device="cuda", dtype="float32"))

    array = backend.asfloat([1, 2, 3])
    result = backend.to_numpy(array)

    assert result.tolist() == [1.0, 2.0, 3.0]
    assert str(result.dtype) == "float32"


def test_gpu_backend_synchronize() -> None:
    backend = make_backend(BackendConfig(device="cuda", dtype="float32"))

    backend.synchronize()


def test_gpu_backend_memory_info() -> None:
    backend = make_backend(BackendConfig(device="cuda", dtype="float32"))

    info = backend.memory_info()

    assert isinstance(info, dict)
    assert "used_bytes" in info
    assert "total_bytes" in info
    assert "pinned_free_blocks" in info


def test_gpu_backend_clear_memory_pool() -> None:
    backend = make_backend(BackendConfig(device="cuda", dtype="float32"))

    backend.clear_memory_pool()
