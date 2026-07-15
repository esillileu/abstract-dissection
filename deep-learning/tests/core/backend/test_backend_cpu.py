from contextlib import AbstractContextManager

import numpy as np

from mlprosection.core.backend import BackendConfig, make_backend


def test_make_cpu_backend_basic_fields() -> None:
    backend = make_backend(BackendConfig(device="cpu", dtype="float32"))

    assert backend.name == "numpy"
    assert backend.device == "cpu"
    assert backend.is_cpu
    assert not backend.is_gpu
    assert backend.dtype_name == "float32"


def test_backend_asarray() -> None:
    backend = make_backend(BackendConfig(device="cpu", dtype="float32"))

    array = backend.asarray([1, 2, 3])

    assert isinstance(array, np.ndarray)
    assert array.tolist() == [1, 2, 3]


def test_backend_asfloat() -> None:
    backend = make_backend(BackendConfig(device="cpu", dtype="float32"))

    array = backend.asfloat([1, 2, 3])

    assert isinstance(array, np.ndarray)
    assert str(array.dtype) == "float32"
    assert array.tolist() == [1.0, 2.0, 3.0]


def test_backend_asint() -> None:
    backend = make_backend(BackendConfig(device="cpu", dtype="float32"))

    array = backend.asint([1.2, 2.8, 3.1])

    assert isinstance(array, np.ndarray)
    assert str(array.dtype) == "int64"
    assert array.tolist() == [1, 2, 3]


def test_to_numpy_on_cpu_returns_array() -> None:
    backend = make_backend(BackendConfig(device="cpu", dtype="float32"))
    array = backend.asfloat([1, 2, 3])

    result = backend.to_numpy(array)

    assert result is array


def test_move_array_from_same_cpu_backend_without_dtype() -> None:
    backend = make_backend(BackendConfig(device="cpu", dtype="float32"))
    array = backend.asfloat([1, 2, 3])

    result = backend.move_array_from(array, backend)

    assert result is array


def test_move_array_from_same_cpu_backend_with_dtype() -> None:
    backend = make_backend(BackendConfig(device="cpu", dtype="float32"))
    array = backend.asarray([1, 2, 3], dtype=backend.int_dtype)

    result = backend.move_array_from(array, backend, dtype=backend.float_dtype)

    assert isinstance(result, np.ndarray)
    assert str(result.dtype) == "float32"
    assert result.tolist() == [1.0, 2.0, 3.0]


def test_scalar_to_float() -> None:
    backend = make_backend(BackendConfig(device="cpu", dtype="float32"))
    scalar = backend.asarray(3.5, dtype=backend.float_dtype)

    assert backend.scalar_to_float(scalar) == 3.5


def test_scalar_to_int() -> None:
    backend = make_backend(BackendConfig(device="cpu", dtype="float32"))
    scalar = backend.asarray(3, dtype=backend.int_dtype)

    assert backend.scalar_to_int(scalar) == 3


def test_seed_is_deterministic_for_cpu_backend() -> None:
    backend = make_backend(BackendConfig(device="cpu", dtype="float32"))

    backend.seed(123)
    first = backend.xp.random.rand(3)

    backend.seed(123)
    second = backend.xp.random.rand(3)

    assert np.array_equal(first, second)


def test_synchronize_on_cpu_is_noop() -> None:
    backend = make_backend(BackendConfig(device="cpu", dtype="float32"))

    backend.synchronize()


def test_range_on_cpu_returns_context_manager() -> None:
    backend = make_backend(BackendConfig(device="cpu", dtype="float32"))

    ctx = backend.range("test")

    assert isinstance(ctx, AbstractContextManager)

    with ctx:
        value = 1

    assert value == 1


def test_memory_info_on_cpu_is_none() -> None:
    backend = make_backend(BackendConfig(device="cpu", dtype="float32"))

    assert backend.memory_info() is None


def test_clear_memory_pool_on_cpu_is_noop() -> None:
    backend = make_backend(BackendConfig(device="cpu", dtype="float32"))

    backend.clear_memory_pool()