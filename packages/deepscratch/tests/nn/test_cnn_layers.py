from __future__ import annotations

import numpy as np
import pytest
from deepscratch.core import Tensor
from deepscratch.nn.layers import MaxPool2D
from deepscratch.nn.utils.cnn import im2col


def _reference_im2col(x, kernel_size, stride, padding, *, pad_value=0):
    kernel_h, kernel_w = kernel_size
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    batch_size, channels, input_h, input_w = x.shape
    output_h = (input_h + 2 * pad_h - kernel_h) // stride_h + 1
    output_w = (input_w + 2 * pad_w - kernel_w) // stride_w + 1
    padded = np.pad(
        x,
        ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)),
        mode="constant",
        constant_values=pad_value,
    )
    rows = []
    for batch in range(batch_size):
        for output_y in range(output_h):
            for output_x in range(output_w):
                window = padded[
                    batch,
                    :,
                    output_y * stride_h : output_y * stride_h + kernel_h,
                    output_x * stride_w : output_x * stride_w + kernel_w,
                ]
                rows.append(window.reshape(channels * kernel_h * kernel_w))
    return np.asarray(rows)


@pytest.mark.parametrize(
    ("kernel_size", "stride", "padding"),
    [
        ((2, 2), (1, 1), (0, 0)),
        ((3, 2), (2, 1), (1, 0)),
        ((2, 3), (1, 2), (0, 1)),
    ],
)
def test_im2col_sliding_windows_match_reference(
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
    padding: tuple[int, int],
) -> None:
    x = np.arange(2 * 2 * 5 * 6, dtype=np.float64).reshape(2, 2, 5, 6)

    actual = im2col(
        x,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        xp=np,
        pad_value=-7,
    )

    expected = _reference_im2col(
        x,
        kernel_size,
        stride,
        padding,
        pad_value=-7,
    )
    np.testing.assert_array_equal(actual, expected)


def test_max_pool_reuses_argmax_for_forward_and_backward() -> None:
    layer = MaxPool2D(2, 2)
    x = Tensor(
        np.asarray(
            [
                [
                    [
                        [1.0, 6.0, 2.0, 8.0],
                        [3.0, 4.0, 7.0, 5.0],
                        [9.0, 10.0, 13.0, 12.0],
                        [11.0, 14.0, 15.0, 16.0],
                    ]
                ]
            ],
        )
    )

    output = layer.forward(x)
    dx = layer.backward(Tensor(np.ones_like(output.data)))

    np.testing.assert_array_equal(
        output.data,
        np.asarray([[[[6.0, 8.0], [14.0, 16.0]]]]),
    )
    np.testing.assert_array_equal(
        dx.data,
        np.asarray(
            [
                [
                    [
                        [0.0, 1.0, 0.0, 1.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 1.0],
                    ]
                ]
            ],
        ),
    )
