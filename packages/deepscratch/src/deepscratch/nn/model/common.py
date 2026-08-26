from __future__ import annotations

from deepscratch.nn.initailizer import he_normal_, xavier_normal_
from deepscratch.nn.layers import Affine, Layer, Relu, Sigmoid, Tanh


def activation(name: str) -> Layer:
    if name == "relu":
        return Relu()
    if name == "sigmoid":
        return Sigmoid()
    if name == "tanh":
        return Tanh()
    raise ValueError(f"unknown activation: {name}")


def initialize_affine(layer: Affine, initializer: str) -> None:
    if initializer == "he":
        he_normal_(layer.W)
    elif initializer == "xavier":
        xavier_normal_(layer.W)
    elif initializer.startswith("std:"):
        std = float(initializer.split(":", 1)[1])
        values = layer.backend.random_stream("model_init").randn(*layer.W.shape) * std
        layer.W.data[...] = values.astype(layer.W.dtype, copy=False)
    else:
        raise ValueError(f"unknown initializer: {initializer}")
