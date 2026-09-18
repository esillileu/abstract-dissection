from __future__ import annotations

from typing import Any

from deepscratch.core import Tensor
from deepscratch.core.backend import Backend, get_default_backend, resolve_backend

from ...types import Parameter
from ..base import Layer


def _as_array(value: Tensor | Any, backend: Backend):
    if isinstance(value, Tensor):
        return value.data
    return backend.asarray(value)


def _as_index_array(value: Tensor | Any, backend: Backend):
    xp = backend.xp
    if isinstance(value, Tensor):
        return value.data.astype(xp.int64, copy=False)
    return xp.asarray(value, dtype=xp.int64)


def _sigmoid_array(x, xp):
    return 1 / (1 + xp.exp(-x))


def _make_parameter(
    shape: tuple[int, ...],
    *,
    backend: Backend | str | None,
    scale: float,
    name: str,
    zeros: bool = False,
) -> Parameter:
    resolved = (
        resolve_backend(backend) if backend is not None else get_default_backend()
    )
    xp = resolved.xp
    if zeros:
        data = xp.zeros(shape, dtype=resolved.float_dtype)
    else:
        data = (scale * resolved.random_stream("model_init").randn(*shape)).astype(
            resolved.float_dtype
        )
    return Parameter(data, backend=resolved, name=name)


class TimeLayer(Layer):
    """Base contract for layers whose leading shape is ``(batch, time, ...)``."""

    time_axis = 1

    def reset_state(self) -> None:
        """Reset optional state carried between truncated-BPTT batches."""

    def detach_state(self) -> None:
        """Keep values but clear backward-time caches at a BPTT boundary."""


class RecurrentTimeLayer(TimeLayer):
    """Common state lifecycle for recurrent time layers."""

    def __init__(
        self, *, stateful: bool = False, backend: Backend | str | None = None
    ) -> None:
        super().__init__(backend)
        self.stateful = stateful

    def detach_state(self) -> None:
        layers = getattr(self, "layers", None)
        if isinstance(layers, list):
            layers.clear()
