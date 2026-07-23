from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterator, TypeAlias, TYPE_CHECKING

import numpy as np

from mlprosection.core import Backend, resolve_backend, get_default_backend
from ..types.parameter import Parameter

if TYPE_CHECKING:
    from mlprosection import Tensor

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
        path = Path(path)

        arrays: dict[str, Any] = {}

        for name, param in self.named_parameters():
            arrays[name] = param.backend.to_numpy(param.data).copy()

        np.savez(path, **arrays)

    def load_params_npz(
        self,
        path: str | Path,
        strict: bool = True,
    ) -> None:
        path = Path(path)
        named_params = dict(self.named_parameters())

        with np.load(path, allow_pickle=False) as data:
            saved_names = set(data.files)
            current_names = set(named_params)

            if strict:
                missing = current_names - saved_names
                unexpected = saved_names - current_names

                if missing:
                    raise KeyError(f"missing parameters: {sorted(missing)}")

                if unexpected:
                    raise KeyError(f"unexpected parameters: {sorted(unexpected)}")

            for name, param in named_params.items():
                if name not in data:
                    continue

                array = data[name]

                if array.shape != param.data.shape:
                    raise ValueError(
                        f"shape mismatch for {name!r}: "
                        f"expected {param.data.shape}, got {array.shape}"
                    )

                new_data = param.backend.asarray(array, dtype=param.data.dtype)
                param.data[...] = new_data


def _iter_named_parameters(
    value: Any,
    prefix: str,
    seen: set[int],
) -> Iterator[tuple[str, Parameter]]:
    if isinstance(value, Parameter):
        if not value.requires_grad:
            return

        param_id = id(value)

        if param_id in seen:
            return

        seen.add(param_id)
        yield prefix, value
        return

    if isinstance(value, Layer):
        for name, item in value.__dict__.items():
            child_prefix = f"{prefix}.{name}"
            yield from _iter_named_parameters(
                value=item,
                prefix=child_prefix,
                seen=seen,
            )

        return

    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            item_prefix = f"{prefix}.{index}"
            yield from _iter_named_parameters(
                value=item,
                prefix=item_prefix,
                seen=seen,
            )

        return

    if isinstance(value, dict):
        for key, item in value.items():
            item_prefix = f"{prefix}.{key}"
            yield from _iter_named_parameters(
                value=item,
                prefix=item_prefix,
                seen=seen,
            )


def _iter_layers(value: Any, seen: set[int]) -> Iterator[Layer]:
    if isinstance(value, Layer):
        layer_id = id(value)

        if layer_id in seen:
            return

        seen.add(layer_id)
        yield value
        return

    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_layers(item, seen)

        return

    if isinstance(value, dict):
        for item in value.values():
            yield from _iter_layers(item, seen)


def _iter_named_buffers(
    layer: Layer,
    prefix: str,
    seen_layers: set[int],
    runtime_state: bool | None,
) -> Iterator[tuple[str, Any]]:
    layer_id = id(layer)
    if layer_id in seen_layers:
        return
    seen_layers.add(layer_id)

    for name, is_runtime in getattr(layer, "_buffers", {}).items():
        if runtime_state is None or runtime_state == is_runtime:
            yield f"{prefix}{name}", getattr(layer, name)

    for name, value in layer.__dict__.items():
        child_prefix = f"{prefix}{name}."
        if isinstance(value, Layer):
            yield from _iter_named_buffers(
                value, child_prefix, seen_layers, runtime_state
            )
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                if isinstance(child, Layer):
                    yield from _iter_named_buffers(
                        child,
                        f"{child_prefix}{index}.",
                        seen_layers,
                        runtime_state,
                    )
        elif isinstance(value, dict):
            for key, child in value.items():
                if isinstance(child, Layer):
                    yield from _iter_named_buffers(
                        child,
                        f"{child_prefix}{key}.",
                        seen_layers,
                        runtime_state,
                    )


def _resolve_owner(root: Layer, path: str) -> tuple[Layer, str]:
    parts = path.split(".")
    value: Any = root
    for part in parts[:-1]:
        if isinstance(value, (list, tuple)):
            value = value[int(part)]
        elif isinstance(value, dict):
            value = value[part]
        else:
            value = getattr(value, part)
    if not isinstance(value, Layer):
        raise TypeError(f"buffer owner for {path!r} is not a Layer")
    return value, parts[-1]
