"""2D Max Pooling layer implementation."""

from __future__ import annotations

from typing import Any

from deepscratch.core import Tensor
from deepscratch.nn.layers.base import Layer
from deepscratch.nn.utils.cnn import _calculate_output_size, _pair, col2im, im2col

from .types import IntPair


class MaxPool2D(Layer):
    def __init__(
        self,
        kernel_size: IntPair,
        stride: IntPair | None = None,
        padding: IntPair = 0,
    ) -> None:
        super().__init__()

        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(kernel_size if stride is None else stride)
        self.padding = _pair(padding)

        self._input_shape: tuple[int, int, int, int] | None = None
        self._argmax: Any | None = None

    def forward_manual(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(
                f"MaxPool2D input must have shape (N, C, H, W), got {x.shape}"
            )

        xp = x.backend.xp

        batch_size, channels, input_h, input_w = x.shape
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride
        pad_h, pad_w = self.padding

        output_h = _calculate_output_size(
            input_size=input_h,
            kernel_size=kernel_h,
            stride=stride_h,
            padding=pad_h,
        )
        output_w = _calculate_output_size(
            input_size=input_w,
            kernel_size=kernel_w,
            stride=stride_w,
            padding=pad_w,
        )

        col = im2col(
            x.data,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            xp=xp,
            pad_value=-xp.inf,
        )

        pool_size = kernel_h * kernel_w

        col = col.reshape(
            batch_size * output_h * output_w,
            channels,
            pool_size,
        )

        argmax = xp.argmax(col, axis=2)
        out = xp.take_along_axis(col, argmax[..., None], axis=2)[..., 0]

        out = out.reshape(
            batch_size,
            output_h,
            output_w,
            channels,
        )
        out = xp.transpose(out, (0, 3, 1, 2))

        self._input_shape = x.shape
        self._argmax = argmax

        return Tensor(
            out,
            backend=x.backend,
            requires_grad=x.requires_grad,
        )

    def backward_manual(self, dout: Tensor) -> Tensor:
        if self._input_shape is None or self._argmax is None:
            raise RuntimeError(
                "forward_manual() must be called before backward_manual()"
            )

        xp = dout.backend.xp

        batch_size, channels, _, _ = self._input_shape
        kernel_h, kernel_w = self.kernel_size

        pool_size = kernel_h * kernel_w

        dout_data = xp.transpose(
            dout.data,
            (0, 2, 3, 1),
        ).reshape(-1, channels)

        dcol = xp.zeros(
            (
                dout_data.shape[0],
                channels,
                pool_size,
            ),
            dtype=dout_data.dtype,
        )

        row_indices = xp.arange(dout_data.shape[0])[:, None]
        channel_indices = xp.arange(channels)[None, :]

        dcol[
            row_indices,
            channel_indices,
            self._argmax,
        ] = dout_data

        dcol = dcol.reshape(
            dout_data.shape[0],
            channels * pool_size,
        )

        dx = col2im(
            dcol,
            input_shape=self._input_shape,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            xp=xp,
        )

        return Tensor(
            dx,
            backend=dout.backend,
            requires_grad=False,
        )
