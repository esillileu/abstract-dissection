from __future__ import annotations

from typing import Any

from deepscratch.core import Tensor
from deepscratch.core.backend import Backend

from ..base import Layer
from .base import RecurrentTimeLayer, _as_array, _make_parameter, _sigmoid_array


class GRU(Layer):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        *,
        backend: Backend | str | None = None,
    ) -> None:
        super().__init__(backend)
        self.Wx = _make_parameter(
            (input_size, 3 * hidden_size),
            backend=self._backend,
            scale=1 / input_size**0.5,
            name="Wx",
        )
        self.Wh = _make_parameter(
            (hidden_size, 3 * hidden_size),
            backend=self._backend,
            scale=1 / hidden_size**0.5,
            name="Wh",
        )
        self.cache = None

    def forward_manual(self, x: Tensor, h_prev: Tensor) -> Tensor:
        xp = x.backend.xp
        hidden_size = self.Wh.shape[0]
        wxz, wxr, wx = (
            self.Wx.data[:, :hidden_size],
            self.Wx.data[:, hidden_size : 2 * hidden_size],
            self.Wx.data[:, 2 * hidden_size :],
        )
        whz, whr, wh = (
            self.Wh.data[:, :hidden_size],
            self.Wh.data[:, hidden_size : 2 * hidden_size],
            self.Wh.data[:, 2 * hidden_size :],
        )

        z = _sigmoid_array(x.data @ wxz + h_prev.data @ whz, xp)
        r = _sigmoid_array(x.data @ wxr + h_prev.data @ whr, xp)
        h_hat = xp.tanh(x.data @ wx + (r * h_prev.data) @ wh)
        h_next = (1 - z) * h_prev.data + z * h_hat
        self.cache = (x.data, h_prev.data, z, r, h_hat, x.backend)
        return Tensor(h_next, backend=x.backend)

    def backward_manual(self, dh_next: Tensor) -> tuple[Tensor, Tensor]:
        if self.cache is None:
            raise RuntimeError("forward must be called before backward")

        x, h_prev, z, r, h_hat, backend = self.cache
        xp = backend.xp
        hidden_size = self.Wh.shape[0]
        wxz, wxr, wx = (
            self.Wx.data[:, :hidden_size],
            self.Wx.data[:, hidden_size : 2 * hidden_size],
            self.Wx.data[:, 2 * hidden_size :],
        )
        whz, whr, wh = (
            self.Wh.data[:, :hidden_size],
            self.Wh.data[:, hidden_size : 2 * hidden_size],
            self.Wh.data[:, 2 * hidden_size :],
        )

        dh_hat = dh_next.data * z
        dh_prev = dh_next.data * (1 - z)
        dt = dh_hat * (1 - h_hat**2)
        d_wh = (r * h_prev).T @ dt
        dhr = dt @ wh.T
        d_wx = x.T @ dt
        dx = dt @ wx.T
        dh_prev += r * dhr

        dz = dh_next.data * h_hat - dh_next.data * h_prev
        dt = dz * z * (1 - z)
        d_whz = h_prev.T @ dt
        dh_prev += dt @ whz.T
        d_wxz = x.T @ dt
        dx += dt @ wxz.T

        dr = dhr * h_prev
        dt = dr * r * (1 - r)
        d_whr = h_prev.T @ dt
        dh_prev += dt @ whr.T
        d_wxr = x.T @ dt
        dx += dt @ wxr.T

        self.Wx.grad[...] = xp.hstack((d_wxz, d_wxr, d_wx))
        self.Wh.grad[...] = xp.hstack((d_whz, d_whr, d_wh))
        return Tensor(dx, backend=backend), Tensor(dh_prev, backend=backend)


class TimeGRU(RecurrentTimeLayer):
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
            (input_size, 3 * hidden_size),
            backend=self._backend,
            scale=1 / input_size**0.5,
            name="Wx",
        )
        self.Wh = _make_parameter(
            (hidden_size, 3 * hidden_size),
            backend=self._backend,
            scale=1 / hidden_size**0.5,
            name="Wh",
        )
        self.layers: list[tuple[Any, ...]] = []
        self.register_buffer("h", runtime_state=True)
        self.dh = None

    def forward_manual(self, xs: Tensor) -> Tensor:
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
            wxz, wxr, wx = (
                self.Wx.data[:, :hidden_size],
                self.Wx.data[:, hidden_size : 2 * hidden_size],
                self.Wx.data[:, 2 * hidden_size :],
            )
            whz, whr, wh = (
                self.Wh.data[:, :hidden_size],
                self.Wh.data[:, hidden_size : 2 * hidden_size],
                self.Wh.data[:, 2 * hidden_size :],
            )
            z = _sigmoid_array(x @ wxz + h @ whz, xp)
            r = _sigmoid_array(x @ wxr + h @ whr, xp)
            h_hat = xp.tanh(x @ wx + (r * h) @ wh)
            h_prev = h
            h = (1 - z) * h_prev + z * h_hat
            hs[:, index, :] = h
            self.layers.append((x, h_prev, z, r, h_hat))

        self.h = h
        return Tensor(hs, backend=xs.backend)

    def backward_manual(self, dhs: Tensor) -> Tensor:
        if not self.layers:
            raise RuntimeError("forward must be called before backward")

        xp = dhs.backend.xp
        n, time_size, hidden_size = dhs.shape
        input_size = self.Wx.shape[0]
        dxs = xp.empty((n, time_size, input_size), dtype=dhs.dtype)
        d_wx_total = xp.zeros_like(self.Wx.data)
        d_wh_total = xp.zeros_like(self.Wh.data)
        dh = xp.zeros((n, hidden_size), dtype=dhs.dtype)

        for index in reversed(range(time_size)):
            x, h_prev, z, r, h_hat = self.layers[index]
            wxz, wxr, wx = (
                self.Wx.data[:, :hidden_size],
                self.Wx.data[:, hidden_size : 2 * hidden_size],
                self.Wx.data[:, 2 * hidden_size :],
            )
            whz, whr, wh = (
                self.Wh.data[:, :hidden_size],
                self.Wh.data[:, hidden_size : 2 * hidden_size],
                self.Wh.data[:, 2 * hidden_size :],
            )

            dh_next = dhs.data[:, index, :] + dh
            dh_hat = dh_next * z
            dh = dh_next * (1 - z)
            dt = dh_hat * (1 - h_hat**2)
            d_wh = (r * h_prev).T @ dt
            dhr = dt @ wh.T
            d_wx = x.T @ dt
            dx = dt @ wx.T
            dh += r * dhr

            dz = dh_next * h_hat - dh_next * h_prev
            dt = dz * z * (1 - z)
            d_whz = h_prev.T @ dt
            dh += dt @ whz.T
            d_wxz = x.T @ dt
            dx += dt @ wxz.T

            dr = dhr * h_prev
            dt = dr * r * (1 - r)
            d_whr = h_prev.T @ dt
            dh += dt @ whr.T
            d_wxr = x.T @ dt
            dx += dt @ wxr.T

            d_wx_total += xp.hstack((d_wxz, d_wxr, d_wx))
            d_wh_total += xp.hstack((d_whz, d_whr, d_wh))
            dxs[:, index, :] = dx

        self.Wx.grad[...] = d_wx_total
        self.Wh.grad[...] = d_wh_total
        self.dh = Tensor(dh, backend=dhs.backend)
        return Tensor(dxs, backend=dhs.backend)

    def set_state(self, h: Tensor | Any) -> None:
        self.h = _as_array(h, self.backend)

    def reset_state(self) -> None:
        self.h = None
