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
    def __init__(self, backend: Backend | None = None) -> Backend:
        self._backend = backend or get_default_backend()


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

    def children(self) -> Iterator[Layer]:
        seen: set[int] = set()

        for value in self.__dict__.values():
            yield from _iter_layers(value, seen)

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
