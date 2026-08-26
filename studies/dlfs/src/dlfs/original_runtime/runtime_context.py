"""Per-run controls used by the promoted original-source domains."""

from __future__ import annotations

from contextvars import ContextVar

_MASTER_SEED: ContextVar[int] = ContextVar("original_master_seed", default=1)
_DEVICE: ContextVar[str] = ContextVar("original_device", default="cuda:0")
_CONFIG: ContextVar[dict[str, object]] = ContextVar("original_config", default={})


def master_seed() -> int:
    return _MASTER_SEED.get()


def device() -> str:
    return _DEVICE.get()


def budget(name: str, default: int) -> int:
    training = _CONFIG.get().get("training", {})
    if isinstance(training, dict) and training.get(name) is not None:
        return int(training[name])
    return default


def set_runtime(*, seed: int, selected_device: str, config: dict[str, object]):
    return (
        _MASTER_SEED.set(seed),
        _DEVICE.set(selected_device),
        _CONFIG.set(config),
    )


def reset_runtime(tokens: tuple[object, object, object]) -> None:
    _MASTER_SEED.reset(tokens[0])  # type: ignore[arg-type]
    _DEVICE.reset(tokens[1])  # type: ignore[arg-type]
    _CONFIG.reset(tokens[2])  # type: ignore[arg-type]


def array_module():
    if device().startswith("cuda"):
        import cupy

        index = int(device().split(":", 1)[1])
        cupy.cuda.Device(index).use()
        return cupy
    import numpy

    return numpy


def synchronize() -> None:
    if device().startswith("cuda"):
        array_module().cuda.get_current_stream().synchronize()
