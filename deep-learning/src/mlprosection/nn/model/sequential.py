from __future__ import annotations

from typing import Iterator

from ..layers.base import Layer


class Sequential(Layer):
    def __init__(self, *layers: Layer, backend=None) -> None:
        super().__init__(backend or (layers[0].backend if layers else None))
        self.layers = list(layers)

    def forward_manual(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward_manual(self, dout):
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def forward_auto(self, x):
        for layer in self.layers:
            x = layer.forward_auto(x)
        return x

    def __iter__(self) -> Iterator[Layer]:
        return iter(self.layers)

    def __len__(self) -> int:
        return len(self.layers)

    def __getitem__(self, idx: int) -> Layer:
        return self.layers[idx]

    def append(self, layer: Layer) -> None:
        self.layers.append(layer)
