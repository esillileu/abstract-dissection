from __future__ import annotations

from mlprosection import Tensor

from .base import Layer


class Flatten(Layer):
    def __init__(self, start_dim: int = 1) -> None:
        super().__init__()
        self.start_dim = start_dim
        self._input_shape: tuple[int, ...] | None = None

    def forward_manual(self, x: Tensor) -> Tensor:
        self._input_shape = x.shape

        if self.start_dim != 1:
            raise NotImplementedError(
                "manual Flatten currently supports start_dim=1 only"
            )

        batch_size = x.shape[0]

        return x.reshape(batch_size, -1)

    def backward_manual(self, dout: Tensor) -> Tensor:
        if self._input_shape is None:
            raise RuntimeError(
                "forward_manual() must be called before backward_manual()"
            )

        return dout.reshape(self._input_shape)