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
    conv_channels = tuple(channel for channel in channels for _ in range(2))
    paddings = (1, 1, 1, 2, 1, 1)
    for index, (out_channels, padding) in enumerate(zip(conv_channels, paddings, strict=True), start=1):
        layers.extend((Conv2D(in_channels, out_channels, (3, 3), 1, padding, initializer=initializer), Relu()))
        in_channels = out_channels
        spatial = spatial - 3 + 2 * padding + 1
        if index % 2 == 0:
            layers.append(MaxPool2D(2, 2))
            spatial //= 2
    hidden = Affine(in_channels * spatial * spatial, hidden_size)
    output = Affine(hidden_size, num_classes)
    initialize_affine(hidden, initializer)
    initialize_affine(output, initializer)
    layers.extend((Flatten(), hidden, Relu(), Dropout(dropout_ratio), output, Dropout(dropout_ratio)))
    return Sequential(*layers)
