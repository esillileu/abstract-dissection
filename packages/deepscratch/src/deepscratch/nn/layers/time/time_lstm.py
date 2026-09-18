from __future__ import annotations

from typing import Any

from deepscratch.core import Tensor
from deepscratch.core.backend import Backend

from .base import RecurrentTimeLayer, _as_array, _make_parameter, _sigmoid_array
from .time_lstm_cuda import backward_cuda_float32, forward_cuda_float32


class TimeLSTM(RecurrentTimeLayer):
    _fused_cuda = True

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        *,
        stateful: bool = False,
        backend: Backend | str | None = None,
    ) -> None:
        super().__init__(stateful=stateful, backend=backend)
        self.Wx = _make_parameter(
            (input_size, 4 * hidden_size),
            backend=self._backend,
            scale=1 / input_size**0.5,
            name="Wx",
        )
        self.Wh = _make_parameter(
            (hidden_size, 4 * hidden_size),
            backend=self._backend,
            scale=1 / hidden_size**0.5,
            name="Wh",
        )
        self.b = _make_parameter(
            (4 * hidden_size,),
            backend=self._backend,
            scale=1.0,
            name="b",
            zeros=True,
        )
        self.layers: list[tuple[Any, ...]] = []
        self.register_buffer("h", runtime_state=True)
        self.register_buffer("c", runtime_state=True)
        self.dh = None

    def forward_manual(self, xs: Tensor, *, cache: bool = True) -> Tensor:
        if (
            self._fused_cuda
            and self.backend.is_gpu
            and xs.dtype == xs.backend.xp.float32
        ):
            return self._forward_cuda_float32(xs, cache=cache)
        xp = xs.backend.xp
        n, time_size, input_size = xs.shape
        hidden_size = self.Wh.shape[0]

        if not self.stateful or self.h is None or self.h.shape[0] != n:
            self.h = xp.zeros((n, hidden_size), dtype=xs.dtype)
        if not self.stateful or self.c is None or self.c.shape[0] != n:
            self.c = xp.zeros((n, hidden_size), dtype=xs.dtype)

        # The input projection is independent across timesteps.  Flattening it
        # into one GEMM avoids T small launches while preserving the recurrent
        # h @ Wh dependency in the loop.
        x_flat = xp.ascontiguousarray(xs.data.reshape(n * time_size, input_size))
        with self.backend.range("TimeLSTM/forward_input_gemm"):
            input_projection = (x_flat @ self.Wx.data).reshape(
                n, time_size, 4 * hidden_size
            )
        hs = xp.empty((n, time_size, hidden_size), dtype=xs.dtype)
        self.layers = []
        h = self.h
        c = self.c

        if cache:
            h_prev_sequence = xp.empty_like(hs)
            c_prev_sequence = xp.empty_like(hs)
            gates = xp.empty((n, time_size, 4 * hidden_size), dtype=xs.dtype)
            cells = xp.empty_like(hs)

        with self.backend.range("TimeLSTM/forward_recurrent_loop"):
            for index in range(time_size):
                h_prev = h
                c_prev = c
                recurrent_projection = h_prev @ self.Wh.data
                a = input_projection[:, index, :] + recurrent_projection + self.b.data

                f = _sigmoid_array(a[:, :hidden_size], xp)
                g = xp.tanh(a[:, hidden_size : 2 * hidden_size])
                i = _sigmoid_array(a[:, 2 * hidden_size : 3 * hidden_size], xp)
                o = _sigmoid_array(a[:, 3 * hidden_size :], xp)

                c = f * c_prev + g * i
                h = o * xp.tanh(c)
                hs[:, index, :] = h
                if cache:
                    h_prev_sequence[:, index, :] = h_prev
                    c_prev_sequence[:, index, :] = c_prev
                    gates[:, index, :hidden_size] = f
                    gates[:, index, hidden_size : 2 * hidden_size] = g
                    gates[:, index, 2 * hidden_size : 3 * hidden_size] = i
                    gates[:, index, 3 * hidden_size :] = o
                    cells[:, index, :] = c

        if cache:
            self.layers.append(
                (
                    x_flat,
                    h_prev_sequence,
                    c_prev_sequence,
                    gates,
                    cells,
                )
            )

        self.h = h
        self.c = c
        return Tensor(hs, backend=xs.backend)

    def backward_manual(self, dhs: Tensor) -> Tensor:
        if (
            self._fused_cuda
            and self.backend.is_gpu
            and dhs.dtype == dhs.backend.xp.float32
        ):
            return self._backward_cuda_float32(dhs)
        if not self.layers:
            raise RuntimeError("forward must be called before backward")

        xp = dhs.backend.xp
        n, time_size, hidden_size = dhs.shape
        input_size = self.Wx.shape[0]
        x_flat, h_prev_sequence, c_prev_sequence, gates, cells = self.layers[0]
        da_sequence = xp.empty((n, time_size, 4 * hidden_size), dtype=dhs.dtype)
        dh = xp.zeros((n, hidden_size), dtype=dhs.dtype)
        dc = xp.zeros((n, hidden_size), dtype=dhs.dtype)

        with self.backend.range("TimeLSTM/backward_recurrent_loop"):
            for index in reversed(range(time_size)):
                c_prev = c_prev_sequence[:, index, :]
                c_next = cells[:, index, :]
                f = gates[:, index, :hidden_size]
                g = gates[:, index, hidden_size : 2 * hidden_size]
                i = gates[:, index, 2 * hidden_size : 3 * hidden_size]
                o = gates[:, index, 3 * hidden_size :]
                tanh_c_next = xp.tanh(c_next)
                upstream = dhs.data[:, index, :] + dh
                ds = dc + upstream * o * (1 - tanh_c_next**2)
                dc = ds * f
                di = ds * g
                df = ds * c_prev
                do = upstream * tanh_c_next
                dg = ds * i

                di *= i * (1 - i)
                df *= f * (1 - f)
                do *= o * (1 - o)
                dg *= 1 - g**2

                da = da_sequence[:, index, :]
                da[:, :hidden_size] = df
                da[:, hidden_size : 2 * hidden_size] = dg
                da[:, 2 * hidden_size : 3 * hidden_size] = di
                da[:, 3 * hidden_size :] = do
                dh = da @ self.Wh.data.T

        da_flat = da_sequence.reshape(n * time_size, 4 * hidden_size)
        h_prev_flat = h_prev_sequence.reshape(n * time_size, hidden_size)
        with self.backend.range("TimeLSTM/backward_dWx_gemm"):
            self.Wx.grad[...] = x_flat.T @ da_flat
        with self.backend.range("TimeLSTM/backward_dWh_gemm"):
            self.Wh.grad[...] = h_prev_flat.T @ da_flat
        self.b.grad[...] = da_flat.sum(axis=0)
        with self.backend.range("TimeLSTM/backward_dX_gemm"):
            dxs = (da_flat @ self.Wx.data.T).reshape(n, time_size, input_size)
        self.dh = Tensor(dh, backend=dhs.backend)
        return Tensor(dxs, backend=dhs.backend)

    def _forward_cuda_float32(self, xs: Tensor, *, cache: bool) -> Tensor:
        return forward_cuda_float32(self, xs, cache=cache)

    def _backward_cuda_float32(self, dhs: Tensor) -> Tensor:
        return backward_cuda_float32(self, dhs)

    def set_state(self, h: Tensor | Any, c: Tensor | Any | None = None) -> None:
        self.h = _as_array(h, self.backend)
        self.c = None if c is None else _as_array(c, self.backend)

    def reset_state(self) -> None:
        self.h = None
        self.c = None
