from __future__ import annotations

from typing import TYPE_CHECKING

from .sequential import Sequential
from ..layers import (
    Relu,
    Dropout,
    BatchNormalization,
    Affine,
    Conv2D,
    MaxPool2D,
    Flatten,
)

if TYPE_CHECKING:
    from ..types import Activation


def TwoLayerNet(
    input_size=784, hidden_size=50, output_size=10, activation: Activation = Relu
):
    return Sequential(
        Affine(input_size, hidden_size),
        activation(),
        BatchNormalization(),
        Dropout(),
        Affine(hidden_size, output_size),
    )


def SimpleCNN(
    input_dim=(1, 28, 28),
    conv_param={"filter_num": 30, "filter_size": 5, "pad": 0, "stride": 1},
    hidden_size=100,
    output_size=10,
):
    filter_num = conv_param["filter_num"]
    filter_size = conv_param["filter_size"]
    filter_pad = conv_param["pad"]
    filter_stride = conv_param["stride"]
    conv_output_size = (input_dim[1] - filter_size + 2 * filter_pad) / filter_stride + 1
    pool_output_size = int(filter_num * (conv_output_size / 2) * (conv_output_size / 2))
    return Sequential(
        Conv2D(
            input_dim[0],
            filter_num,
            (filter_size, filter_size),
            filter_stride,
            filter_pad,
        ),
        Relu(),
        MaxPool2D(2, 2),
        Flatten(),
        Affine(pool_output_size, hidden_size),
        Relu(),
        Affine(hidden_size, output_size),
    )
