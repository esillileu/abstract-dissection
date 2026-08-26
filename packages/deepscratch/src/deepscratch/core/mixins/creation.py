from __future__ import annotations

from typing import TYPE_CHECKING

from deepscratch.core.backend import get_default_backend, resolve_backend

if TYPE_CHECKING:
    from deepscratch.core.backend import Backend

    from ..base import Tensor


class TensorCreationMixin:
    @classmethod
    def arange(
        cls,
        start: float,
        stop: float | None = None,
        step: float = 1,
        *,
        backend: Backend | None = None,
        requires_grad: bool = False,
    ) -> Tensor:
        backend = backend or get_default_backend()

        if stop is None:
            start, stop = 0, start

        data = backend.xp.arange(start, stop, step)

        return cls(
            data,
            backend=backend,
            requires_grad=requires_grad,
        )

    @classmethod
    def randn(
        cls,
        *shape: int,
        backend: Backend | None = None,
        requires_grad: bool = False,
    ) -> Tensor:
        backend = backend or get_default_backend()

        data = backend.xp.random.randn(*shape)

        return cls(
            data,
            backend=backend,
            requires_grad=requires_grad,
        )

    @classmethod
    def zeros_like(
        cls,
        other: Tensor,
        *,
        backend: Backend | str | None = None,
        requires_grad: bool = False,
        name: str | None = None,
    ) -> Tensor:
        resolved_backend = (
            other.backend if backend is None else resolve_backend(backend)
        )

        data = resolved_backend.xp.zeros(
            other.shape,
            dtype=other.dtype,
        )

        return cls(
            data,
            backend=resolved_backend,
            requires_grad=requires_grad,
            name=name,
        )

    @classmethod
    def zeros_(
        cls,
        *shape: int,
        backend: Backend | str | None = None,
        requires_grad: bool = False,
        name: str | None = None,
    ) -> Tensor:
        backend = backend or get_default_backend()

        data = backend.xp.zeros(*shape)

        return cls(
            data,
            backend=backend,
            requires_grad=requires_grad,
            name=name,
        )
