from __future__ import annotations

from typing import Any, TYPE_CHECKING

from ...backend import assert_same_device

if TYPE_CHECKING:
    from ..base import Tensor


class TensorIndexingMixin:
    def __getitem__(self: Tensor, idx: Any) -> Tensor:
        from ..base import Tensor

        if isinstance(idx, tuple):
            idx = tuple(item.data if isinstance(item, Tensor) else item for item in idx)
        elif isinstance(idx, Tensor):
            idx = idx.data
        return type(self)(
            self.data[idx],
            backend=self.backend,
            requires_grad=self.requires_grad,
        )

    def __setitem__(self: Tensor, idx: Any, value: Any) -> None:
        from ..base import Tensor

        if isinstance(value, Tensor):
            assert_same_device(self, value)

        if isinstance(idx, tuple):
            idx = tuple(
                item.data if isinstance(item, Tensor) else item for item in idx
            )
        elif isinstance(idx, Tensor):
            idx = idx.data

        if isinstance(value, Tensor):
            value = value.data

        self.data[idx] = value
