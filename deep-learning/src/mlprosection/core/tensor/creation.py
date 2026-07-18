from __future__ import annotations

from typing import Any

from ..backend import Backend, get_default_backend
from .base import Tensor


def as_tensor(x: Any, backend: Backend | str | None = None) -> Tensor:
    if isinstance(x, Tensor):
        return x

    if backend is None:
        backend = get_default_backend()

    return Tensor(x, backend=backend)


def tensor(
    data: Any,
    backend: Backend | str | None = None,
    requires_grad: bool = False,
    name: str | None = None,
) -> Tensor:
    return Tensor(
        data=data,
        backend=backend,
        requires_grad=requires_grad,
        name=name,
    )
