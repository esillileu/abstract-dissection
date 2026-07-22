from __future__ import annotations

from typing import Any

from mlprosection import Tensor
from mlprosection.core import Backend
from mlprosection.nn.types.parameter import Parameter

from .base import Layer
from ..utils.cnn import col2im, im2col, _pair, _calculate_output_size

IntPair = int | tuple[int, int]


class Conv2D(Layer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: IntPair,
        stride: IntPair = 1,
        padding: IntPair = 0,
        bias: bool = True,
        initializer: str = "he",
        backend: Backend | None = None,
    ) -> None:
        super().__init__(backend)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)

        kernel_h, kernel_w = self.kernel_size
        fan_in = in_channels * kernel_h * kernel_w
        if initializer == "he":
            scale = (2.0 / fan_in) ** 0.5
        elif initializer.startswith("std:"):
            scale = float(initializer.split(":", 1)[1])
        else:
            raise ValueError(f"unknown convolution initializer: {initializer}")

        weight = self.backend.xp.random.randn(
            out_channels,
            in_channels,
            kernel_h,
            kernel_w,
        )
        weight = self.backend.asarray(
            weight * scale,
            dtype=self.backend.float_dtype,
        )

        self.W = Parameter(
            weight,
            backend=self.backend,
            requires_grad=True,
        )

        if bias:
            bias_data = self.backend.xp.zeros(
                out_channels,
                dtype=self.backend.float_dtype,
            )

            self.b: Parameter | None = Parameter(
                bias_data,
                backend=self.backend,
                requires_grad=True,
            )
        else:
            self.b = None

        self._input_shape: tuple[int, int, int, int] | None = None
        self._col: Any | None = None
        self._col_W: Any | None = None

    def forward_manual(self, x: Tensor) -> Tensor:
        self._validate_input(x)

        xp = x.backend.xp

        batch_size, _, input_h, input_w = x.shape
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
        )

        col_W = self.W.data.reshape(self.out_channels, -1).T

        out = col @ col_W

        if self.b is not None:
            out += self.b.data

        out = out.reshape(
            batch_size,
            output_h,
            output_w,
            self.out_channels,
        )
        out = xp.transpose(out, (0, 3, 1, 2))

        self._input_shape = x.shape
        self._col = col
        self._col_W = col_W

        requires_grad = (
            x.requires_grad
            or self.W.requires_grad
            or (
                self.b is not None
                and self.b.requires_grad
            )
        )

        return Tensor(
            out,
            backend=x.backend,
            requires_grad=requires_grad,
        )

    def backward_manual(self, dout: Tensor) -> Tensor:
        if self._input_shape is None:
            raise RuntimeError(
                "forward_manual() must be called before backward_manual()"
            )

        if self._col is None or self._col_W is None:
            raise RuntimeError("Conv2D forward cache is missing")

        if dout.backend.device != self.backend.device:
            raise ValueError(
                "dout and Conv2D parameters must be on the same device"
            )

        xp = dout.backend.xp

        dout_data = xp.transpose(
            dout.data,
            (0, 2, 3, 1),
        ).reshape(-1, self.out_channels)

        dW = self._col.T @ dout_data
        dW = dW.T.reshape(self.W.data.shape)

        self._set_grad(self.W, dW)

        if self.b is not None:
            db = dout_data.sum(axis=0)
            self._set_grad(self.b, db)

        dcol = dout_data @ self._col_W.T

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

    def _validate_input(self, x: Tensor) -> None:
        if x.ndim != 4:
            raise ValueError(
                "Conv2D input must have shape (N, C, H, W), "
                f"got {x.shape}"
            )

        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"expected {self.in_channels} input channels, "
                f"got {x.shape[1]}"
            )

        if x.backend.device != self.backend.device:
            raise ValueError(
                "input and Conv2D parameters must be on the same device"
            )

    @staticmethod
    def _set_grad(param: Parameter, grad: Any) -> None:
        if param.grad is None:
            param.grad = param.backend.xp.zeros_like(param.data)

        param.grad[...] = grad



class MaxPool2D(Layer):
    def __init__(
        self,
        kernel_size: IntPair,
        stride: IntPair | None = None,
        padding: IntPair = 0,
    ) -> None:
        super().__init__()

        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(
            kernel_size if stride is None else stride
        )
        self.padding = _pair(padding)

        self._input_shape: tuple[int, int, int, int] | None = None
        self._argmax: Any | None = None

    def forward_manual(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(
                "MaxPool2D input must have shape (N, C, H, W), "
                f"got {x.shape}"
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
