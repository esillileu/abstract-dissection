from __future__ import annotations

from mlprosection.nn.layers import Affine, Conv2D, Flatten, MaxPool2D, Relu

from ..common import initialize_affine
from ..sequential import Sequential


def SimpleCNN(*, input_channels: int = 1, image_size: int = 28, num_classes: int = 10, conv_channels: int = 30, kernel_size: int = 5, stride: int = 1, padding: int = 0, hidden_size: int = 100, initializer: str = "std:0.01") -> Sequential:
    conv_size = (image_size - kernel_size + 2 * padding) // stride + 1
    if conv_size <= 0 or conv_size % 2:
        raise ValueError("convolution output must be positive and divisible by pooling size")
    hidden = Affine(conv_channels * (conv_size // 2) ** 2, hidden_size)
    output = Affine(hidden_size, num_classes)
    initialize_affine(hidden, initializer)
    initialize_affine(output, initializer)
    return Sequential(Conv2D(input_channels, conv_channels, (kernel_size, kernel_size), stride, padding, initializer=initializer), Relu(), MaxPool2D(2, 2), Flatten(), hidden, Relu(), output)
