"""A state-aware sequential container for time-series layers."""

from __future__ import annotations

from typing import Iterator

from mlprosection.nn.layers import Layer


class TimeSequential(Layer):
    """Compose time layers and expose a single state-reset boundary."""

    def __init__(self, *layers: Layer) -> None:
        super().__init__(layers[0].backend if layers else None)
        self.layers = list(layers)

    def forward_manual(self, value):
        for layer in self.layers:
            value = layer.forward(value)
        return value

    def backward_manual(self, gradient):
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)
        return gradient

    def reset_state(self) -> None:
        for layer in self.layers:
            reset = getattr(layer, "reset_state", None)
            if reset is not None:
                reset()

    def __iter__(self) -> Iterator[Layer]:
        return iter(self.layers)

    def __len__(self) -> int:
        return len(self.layers)
