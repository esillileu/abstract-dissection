from __future__ import annotations

from collections.abc import Sequence

from mlprosection.nn.layers import Affine, Conv2D, Dropout, Flatten, Layer, MaxPool2D, Relu

from ..common import initialize_affine
from ..sequential import Sequential


def DeepCNN(*, input_channels: int = 1, image_size: int = 28, num_classes: int = 10, channels: Sequence[int] = (16, 32, 64), hidden_size: int = 50, dropout_ratio: float = 0.5, initializer: str = "he") -> Sequential:
    if len(channels) != 3:
        raise ValueError("DeepCNN requires three channel stages")
    layers: list[Layer] = []
    in_channels = input_channels
    spatial = image_size
    for out_channels in channels:
        for _ in range(2):
            layers.extend((Conv2D(in_channels, out_channels, (3, 3), 1, 1), Relu()))
            in_channels = out_channels
        layers.append(MaxPool2D(2, 2))
        spatial //= 2
    hidden = Affine(in_channels * spatial * spatial, hidden_size)
    output = Affine(hidden_size, num_classes)
    initialize_affine(hidden, initializer)
    initialize_affine(output, initializer)
    layers.extend((Flatten(), hidden, Relu(), Dropout(dropout_ratio), output))
    return Sequential(*layers)
