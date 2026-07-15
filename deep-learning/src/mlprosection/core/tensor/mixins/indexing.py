from __future__ import annotations

from typing import Any, TYPE_CHECKING

from mlprosection.core.backend import assert_same_device

if TYPE_CHECKING:
    from mlprosection.core.tensor.base import Tensor


class TensorIndexingMixin:
    def __getitem__(self, idx: Any) -> Tensor:
        return type(self)(
            self.data[idx],
            backend=self.backend,
            requires_grad=self.requires_grad,
        )

    def __setitem__(self, idx: Any, value: Any) -> None:
        if isinstance(value, type(self)):
            assert_same_device(self, value)
            value = value.data

        self.data[idx] = value
