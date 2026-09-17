from __future__ import annotations

from deepscratch.core import Tensor
from deepscratch.core.backend import Backend

from ..base import Layer
from .base import _make_parameter, _sigmoid_array


class LSTM(Layer):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        *,
        backend: Backend | str | None = None,
    ) -> None:
        super().__init__(backend)
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
        self.cache = None

    def forward_manual(
        self,
        x: Tensor,
        h_prev: Tensor,
        c_prev: Tensor,
    ) -> tuple[Tensor, Tensor]:
        xp = x.backend.xp
        hidden_size = h_prev.shape[1]
        a = x.data @ self.Wx.data + h_prev.data @ self.Wh.data + self.b.data

        f = _sigmoid_array(a[:, :hidden_size], xp)
        g = xp.tanh(a[:, hidden_size : 2 * hidden_size])
        i = _sigmoid_array(a[:, 2 * hidden_size : 3 * hidden_size], xp)
        o = _sigmoid_array(a[:, 3 * hidden_size :], xp)

        c_next = f * c_prev.data + g * i
        h_next = o * xp.tanh(c_next)

        self.cache = (x.data, h_prev.data, c_prev.data, i, f, g, o, c_next, x.backend)
        return Tensor(h_next, backend=x.backend), Tensor(c_next, backend=x.backend)

    def backward_manual(
        self,
        dh_next: Tensor,
        dc_next: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if self.cache is None:
            raise RuntimeError("forward must be called before backward")

        x, h_prev, c_prev, i, f, g, o, c_next, backend = self.cache
        xp = backend.xp
        tanh_c_next = xp.tanh(c_next)
        ds = dc_next.data + (dh_next.data * o) * (1 - tanh_c_next**2)

        dc_prev = ds * f
        di = ds * g
        df = ds * c_prev
        do = dh_next.data * tanh_c_next
        dg = ds * i

        di *= i * (1 - i)
        df *= f * (1 - f)
        do *= o * (1 - o)
        dg *= 1 - g**2

        da = xp.hstack((df, dg, di, do))
        self.Wh.grad[...] = h_prev.T @ da
        self.Wx.grad[...] = x.T @ da
        self.b.grad[...] = da.sum(axis=0)

        dx = Tensor(da @ self.Wx.data.T, backend=backend)
        dh_prev = Tensor(da @ self.Wh.data.T, backend=backend)
        dc_prev = Tensor(dc_prev, backend=backend)
        return dx, dh_prev, dc_prev
