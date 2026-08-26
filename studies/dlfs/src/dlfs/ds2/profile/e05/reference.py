"""Frozen pre-Phase-1 TimeLSTM used as the e05 correctness reference."""

from __future__ import annotations

from deepscratch.core import Tensor
from deepscratch.nn.layers.time import TimeLSTM, _sigmoid_array


class ReferenceTimeLSTM(TimeLSTM):
    """The timestep-at-a-time implementation present at baseline commit 8e29004."""

    def forward_manual(self, xs: Tensor, *, cache: bool = True) -> Tensor:
        xp = xs.backend.xp
        n, time_size, _ = xs.shape
        hidden_size = self.Wh.shape[0]

        if not self.stateful or self.h is None or self.h.shape[0] != n:
            self.h = xp.zeros((n, hidden_size), dtype=xs.dtype)
        if not self.stateful or self.c is None or self.c.shape[0] != n:
            self.c = xp.zeros((n, hidden_size), dtype=xs.dtype)

        hs = xp.empty((n, time_size, hidden_size), dtype=xs.dtype)
        self.layers = []
        h, c = self.h, self.c
        with self.backend.range("TimeLSTM/forward_recurrent_loop"):
            for index in range(time_size):
                x = xs.data[:, index, :]
                h_prev, c_prev = h, c
                with self.backend.range("TimeLSTM/forward_input_gemm"):
                    input_projection = x @ self.Wx.data
                a = input_projection + h_prev @ self.Wh.data + self.b.data
                f = _sigmoid_array(a[:, :hidden_size], xp)
                g = xp.tanh(a[:, hidden_size : 2 * hidden_size])
                i = _sigmoid_array(a[:, 2 * hidden_size : 3 * hidden_size], xp)
                o = _sigmoid_array(a[:, 3 * hidden_size :], xp)
                c = f * c_prev + g * i
                h = o * xp.tanh(c)
                hs[:, index, :] = h
                if cache:
                    self.layers.append((x, h_prev, c_prev, i, f, g, o, c))
        self.h, self.c = h, c
        return Tensor(hs, backend=xs.backend)

    def backward_manual(self, dhs: Tensor) -> Tensor:
        if not self.layers:
            raise RuntimeError("forward must be called before backward")
        xp = dhs.backend.xp
        n, time_size, hidden_size = dhs.shape
        dxs = xp.empty((n, time_size, self.Wx.shape[0]), dtype=dhs.dtype)
        d_wx = xp.zeros_like(self.Wx.data)
        d_wh = xp.zeros_like(self.Wh.data)
        db = xp.zeros_like(self.b.data)
        dh = xp.zeros((n, hidden_size), dtype=dhs.dtype)
        dc = xp.zeros((n, hidden_size), dtype=dhs.dtype)
        with self.backend.range("TimeLSTM/backward_recurrent_loop"):
            for index in reversed(range(time_size)):
                x, h_prev, c_prev, i, f, g, o, c_next = self.layers[index]
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
                da = xp.hstack((df, dg, di, do))
                with self.backend.range("TimeLSTM/backward_dWh_gemm"):
                    d_wh += h_prev.T @ da
                with self.backend.range("TimeLSTM/backward_dWx_gemm"):
                    d_wx += x.T @ da
                db += da.sum(axis=0)
                with self.backend.range("TimeLSTM/backward_dX_gemm"):
                    dxs[:, index, :] = da @ self.Wx.data.T
                dh = da @ self.Wh.data.T
        self.Wx.grad[...] = d_wx
        self.Wh.grad[...] = d_wh
        self.b.grad[...] = db
        self.dh = Tensor(dh, backend=dhs.backend)
        return Tensor(dxs, backend=dhs.backend)


def replace_better_rnnlm_lstms(model) -> None:
    """Replace BetterRnnlm LSTMs with reference layers without changing values."""
    for index in (2, 4):
        old = model.layers[index]
        layer = ReferenceTimeLSTM(
            old.Wx.shape[0], old.Wh.shape[0], stateful=True, backend=old.backend
        )
        layer.Wx.data[...] = old.Wx.data
        layer.Wh.data[...] = old.Wh.data
        layer.b.data[...] = old.b.data
        model.layers[index] = layer
    model.lstm_layers = [model.layers[2], model.layers[4]]
