from __future__ import annotations

from typing import Any

from deepscratch.core import Tensor
from deepscratch.core.backend import Backend

from ..base import Layer
from .base import RecurrentTimeLayer, _as_array, _make_parameter


class RNN(Layer):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        *,
        backend: Backend | str | None = None,
        weight_scale: float | None = None,
    ) -> None:
        super().__init__(backend)
        scale = weight_scale if weight_scale is not None else 1 / input_size**0.5
        self.Wx = _make_parameter(
            (input_size, hidden_size),
            backend=self._backend,
            scale=scale,
            name="Wx",
        )
        self.Wh = _make_parameter(
            (hidden_size, hidden_size),
            backend=self._backend,
            scale=1 / hidden_size**0.5,
            name="Wh",
        )
        self.b = _make_parameter(
            (hidden_size,),
            backend=self._backend,
            scale=1.0,
            name="b",
            zeros=True,
        )
        self.cache = None

    def forward_manual(self, x: Tensor, h_prev: Tensor) -> Tensor:
        t = h_prev @ self.Wh + x @ self.Wx + self.b
        h_next = Tensor(
            self.backend.xp.tanh(t.data),
            backend=t.backend,
            requires_grad=t.requires_grad,
        )
        self.cache = (x, h_prev, h_next)
        return h_next

    def backward_manual(self, dh_next: Tensor) -> tuple[Tensor, Tensor]:
        if self.cache is None:
            raise RuntimeError("forward must be called before backward")

        x, h_prev, h_next = self.cache
        xp = x.backend.xp

        dt = dh_next.data * (1 - h_next.data**2)
        self.b.grad[...] = xp.sum(dt, axis=0)
        self.Wh.grad[...] = h_prev.data.T @ dt
        self.Wx.grad[...] = x.data.T @ dt

        dx = Tensor(dt @ self.Wx.data.T, backend=x.backend)
        dh_prev = Tensor(dt @ self.Wh.data.T, backend=h_prev.backend)
        return dx, dh_prev


class TimeRNN(RecurrentTimeLayer):
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
            (input_size, hidden_size),
            backend=self._backend,
            scale=1 / input_size**0.5,
            name="Wx",
        )
        self.Wh = _make_parameter(
            (hidden_size, hidden_size),
            backend=self._backend,
            scale=1 / hidden_size**0.5,
            name="Wh",
        )
        self.b = _make_parameter(
            (hidden_size,),
            backend=self._backend,
            scale=1.0,
            name="b",
            zeros=True,
        )
        self.layers: list[tuple[Any, Any, Any]] = []
        self.register_buffer("h", runtime_state=True)
        self.dh = None

    def forward_manual(self, xs: Tensor, *, cache: bool = True) -> Tensor:
        xp = xs.backend.xp
        n, time_size, _ = xs.shape
        hidden_size = self.Wh.shape[0]

        if not self.stateful or self.h is None or self.h.shape[0] != n:
            self.h = xp.zeros((n, hidden_size), dtype=xs.dtype)

        hs = xp.empty((n, time_size, hidden_size), dtype=xs.dtype)
        self.layers = []
        h = self.h

        for index in range(time_size):
            x = xs.data[:, index, :]
            h_prev = h
            h = xp.tanh(h_prev @ self.Wh.data + x @ self.Wx.data + self.b.data)
            hs[:, index, :] = h
            if cache:
                self.layers.append((x, h_prev, h))

        self.h = h
        return Tensor(hs, backend=xs.backend)

    def backward_manual(self, dhs: Tensor) -> Tensor:
        if not self.layers:
            raise RuntimeError("forward must be called before backward")

        xp = dhs.backend.xp
        n, time_size, hidden_size = dhs.shape
        input_size = self.Wx.shape[0]
        dxs = xp.empty((n, time_size, input_size), dtype=dhs.dtype)
        d_wx = xp.zeros_like(self.Wx.data)
        d_wh = xp.zeros_like(self.Wh.data)
        db = xp.zeros_like(self.b.data)
        dh = xp.zeros((n, hidden_size), dtype=dhs.dtype)

        for index in reversed(range(time_size)):
            x, h_prev, h_next = self.layers[index]
            dt = (dhs.data[:, index, :] + dh) * (1 - h_next**2)
            db += xp.sum(dt, axis=0)
            d_wh += h_prev.T @ dt
            dh = dt @ self.Wh.data.T
            d_wx += x.T @ dt
            dxs[:, index, :] = dt @ self.Wx.data.T

        self.Wx.grad[...] = d_wx
        self.Wh.grad[...] = d_wh
        self.b.grad[...] = db
        self.dh = Tensor(dh, backend=dhs.backend)
        return Tensor(dxs, backend=dhs.backend)

    def set_state(self, h: Tensor | Any) -> None:
        self.h = _as_array(h, self.backend)

    def reset_state(self) -> None:
        self.h = None
