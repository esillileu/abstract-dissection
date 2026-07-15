from .base import Backend
from .device import device_index, normalize_device
from .types import ArrayModule, BackendConfig, FloatDTypeName

_backend_cache: dict[tuple[str, FloatDTypeName, bool], Backend] = {}
_default_backend: Backend | None = None


def make_backend(config: BackendConfig | None = None) -> Backend:
    config = config or BackendConfig()
    device = normalize_device(config.device)
    key = (device, config.dtype, config.profile)

    if key in _backend_cache:
        backend = _backend_cache[key]

        if config.seed is not None:
            backend.seed(config.seed)

        return backend

    if device == "cpu":
        import numpy as np

        xp: ArrayModule = np
        float_dtype = xp.float32 if config.dtype == "float32" else xp.float64

        backend = Backend(
            xp=xp,
            name="numpy",
            device="cpu",
            float_dtype=float_dtype,
            int_dtype=xp.int64,
            bool_dtype=xp.bool_,
            profile=config.profile,
        )

    else:
        try:
            import cupy as cp
        except ImportError as e:
            raise RuntimeError(
                "CUDA backend was requested, but cupy is not installed."
            ) from e

        index = device_index(device)
        assert index is not None

        cp.cuda.Device(index).use()

        xp = cp
        float_dtype = xp.float32 if config.dtype == "float32" else xp.float64

        backend = Backend(
            xp=xp,
            name="cupy",
            device=f"cuda:{index}",
            float_dtype=float_dtype,
            int_dtype=xp.int64,
            bool_dtype=xp.bool_,
            profile=config.profile,
        )

    if config.seed is not None:
        backend.seed(config.seed)

    _backend_cache[key] = backend
    return backend


def get_default_backend() -> Backend:
    global _default_backend

    if _default_backend is None:
        _default_backend = make_backend(BackendConfig(device="cpu"))

    return _default_backend


def set_default_backend(backend: Backend | str | BackendConfig) -> Backend:
    global _default_backend

    if isinstance(backend, Backend):
        _default_backend = backend
    elif isinstance(backend, BackendConfig):
        _default_backend = make_backend(backend)
    elif isinstance(backend, str):
        _default_backend = make_backend(BackendConfig(device=backend))
    else:
        raise TypeError(f"unsupported backend type: {type(backend)!r}")

    return _default_backend


def resolve_backend(target: Backend | str | None = None) -> Backend:
    if target is None:
        return get_default_backend()

    if isinstance(target, Backend):
        return target

    if isinstance(target, str):
        return make_backend(BackendConfig(device=target))

    raise TypeError(f"unsupported backend target: {type(target)!r}")