"""Base layer abstract class definition."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias

from deepscratch.core import Backend, get_default_backend, resolve_backend

from ...types.parameter import Parameter
from .serialization import load_layer_params_npz, save_layer_params_npz
from .traversal import (
    _iter_layers,
    _iter_named_buffers,
    _iter_named_parameters,
    _resolve_owner,
)

if TYPE_CHECKING:
    from deepscratch.core import Tensor

NamedParameter: TypeAlias = tuple[str, Parameter]
NamedParameters: TypeAlias = list[NamedParameter]


class Layer(ABC):
    def __init__(self, backend: Backend | None = None) -> None:
        self._backend = backend or get_default_backend()
        self.training = True
        self._buffers: dict[str, bool] = {}

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs) -> Tensor:
        return self.forward_manual(*args, **kwargs)

    def backward(self, *args, **kwargs) -> Tensor:
        return self.backward_manual(*args, **kwargs)

    @abstractmethod
    def forward_manual(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def backward_manual(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError

    def forward_auto(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError

    def named_parameters(self) -> Iterator[tuple[str, Parameter]]:
        seen: set[int] = set()

        for name, value in self.__dict__.items():
            yield from _iter_named_parameters(
                value=value,
                prefix=name,
                seen=seen,
            )

    def register_buffer(
        self,
        name: str,
        value: Any = None,
        *,
        runtime_state: bool = False,
    ) -> None:
        """Register non-parameter state owned by this layer.

        ``runtime_state`` marks ephemeral recurrent state.  Persistent buffers
        such as BatchNorm statistics are registered with the default value.
        """
        if not name.isidentifier() or name.startswith("_"):
            raise ValueError(f"invalid buffer name: {name!r}")
        if isinstance(getattr(self, name, None), Parameter):
            raise ValueError(f"buffer {name!r} is already a parameter")
        self._buffers[name] = runtime_state
        setattr(self, name, value)

    def named_buffers(
        self,
        *,
        runtime_state: bool | None = None,
    ) -> Iterator[tuple[str, Any]]:
        seen_layers: set[int] = set()
        yield from _iter_named_buffers(
            self,
            prefix="",
            seen_layers=seen_layers,
            runtime_state=runtime_state,
        )

    def children(self) -> Iterator[Layer]:
        seen: set[int] = set()

        for value in self.__dict__.values():
            yield from _iter_layers(value, seen)

    def train(self, mode: bool = True) -> Layer:
        """Set training mode recursively for this layer and its children."""
        self.training = mode
        for child in self.children():
            child.train(mode)
        return self

    def eval(self) -> Layer:
        """Set evaluation mode recursively."""
        return self.train(False)

    def zero_grad(self) -> None:
        for _, p in self.named_parameters():
            p.zero_grad()

    @property
    def backend(self) -> Backend:
        for _, p in self.named_parameters():
            return p.backend
        return self._backend

    @property
    def dtype(self) -> str | None:
        for _, param in self.named_parameters():
            return param.dtype

        return None

    @property
    def device(self) -> str:
        for _, p in self.named_parameters():
            return p.device
        return "cpu"

    def to(self, target: Backend | str) -> Layer:
        backend = resolve_backend(target)

        for _, p in self.named_parameters():
            moved = p.to(backend)
            p.data = moved.data
            p.grad = moved.grad
            p.backend = moved.backend
        for name, value in self.named_buffers():
            if value is None:
                continue
            owner, attr = _resolve_owner(self, name)
            setattr(owner, attr, backend.asarray(owner.backend.to_numpy(value)))
        for layer in (self, *self.children()):
            layer._backend = backend
        return self

    def cpu(self) -> Layer:
        return self.to("cpu")

    def gpu(self, device: str = "cuda:0") -> Layer:
        return self.to(device)

    def save_params_npz(self, path: str | Path) -> None:
        save_layer_params_npz(self, path)

    def load_params_npz(
        self,
        path: str | Path,
        strict: bool = True,
    ) -> None:
        load_layer_params_npz(self, path, strict=strict)
