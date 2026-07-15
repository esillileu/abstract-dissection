import pytest

from mlprosection.core.backend import (
    Backend,
    BackendConfig,
    get_default_backend,
    make_backend,
    resolve_backend,
    set_default_backend,
)


def test_make_backend_uses_cache_for_same_config() -> None:
    first = make_backend(BackendConfig(device="cpu", dtype="float32"))
    second = make_backend(BackendConfig(device="cpu", dtype="float32"))

    assert first is second


def test_make_backend_uses_different_cache_entry_for_different_dtype() -> None:
    float32_backend = make_backend(BackendConfig(device="cpu", dtype="float32"))
    float64_backend = make_backend(BackendConfig(device="cpu", dtype="float64"))

    assert float32_backend is not float64_backend
    assert float32_backend.dtype_name == "float32"
    assert float64_backend.dtype_name == "float64"


def test_make_backend_normalizes_device_for_cache_key() -> None:
    first = make_backend(BackendConfig(device="cpu", dtype="float32"))
    second = make_backend(BackendConfig(device=" CPU ", dtype="float32"))

    assert first is second


def test_get_default_backend_returns_backend() -> None:
    backend = get_default_backend()

    assert isinstance(backend, Backend)
    assert backend.is_cpu


def test_set_default_backend_with_backend_instance() -> None:
    backend = make_backend(BackendConfig(device="cpu", dtype="float32"))

    result = set_default_backend(backend)

    assert result is backend
    assert get_default_backend() is backend


def test_set_default_backend_with_config() -> None:
    result = set_default_backend(BackendConfig(device="cpu", dtype="float64"))

    assert result.is_cpu
    assert result.dtype_name == "float64"
    assert get_default_backend() is result


def test_set_default_backend_with_string() -> None:
    result = set_default_backend("cpu")

    assert result.is_cpu
    assert get_default_backend() is result


def test_set_default_backend_rejects_invalid_type() -> None:
    with pytest.raises(TypeError):
        set_default_backend(123)  # type: ignore[arg-type]


def test_resolve_backend_none_returns_default() -> None:
    default = set_default_backend("cpu")

    assert resolve_backend(None) is default


def test_resolve_backend_with_backend_instance() -> None:
    backend = make_backend(BackendConfig(device="cpu", dtype="float32"))

    assert resolve_backend(backend) is backend


def test_resolve_backend_with_string() -> None:
    backend = resolve_backend("cpu")

    assert backend.is_cpu


def test_resolve_backend_rejects_invalid_type() -> None:
    with pytest.raises(TypeError):
        resolve_backend(123)  # type: ignore[arg-type]