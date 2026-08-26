from repro_core.numerics import (
    Array,
    ArrayModule,
    Backend,
    BackendConfig,
    BackendName,
    Device,
    DeviceMismatchError,
    DeviceTarget,
    DType,
    FloatDTypeName,
    HasDevice,
    assert_same_device,
    device_index,
    get_default_backend,
    get_device,
    make_backend,
    normalize_device,
    resolve_backend,
    same_device,
    set_default_backend,
)


def test_backend_public_imports() -> None:
    assert Array is not None
    assert DType is not None
    assert Device is not None
    assert ArrayModule is not None
    assert Backend is not None
    assert BackendConfig is not None
    assert BackendName is not None
    assert DeviceMismatchError is not None
    assert DeviceTarget is not None
    assert FloatDTypeName is not None
    assert HasDevice is not None

    assert callable(assert_same_device)
    assert callable(device_index)
    assert callable(get_default_backend)
    assert callable(get_device)
    assert callable(make_backend)
    assert callable(normalize_device)
    assert callable(resolve_backend)
    assert callable(same_device)
    assert callable(set_default_backend)
