from __future__ import annotations

from typing import Any

from mlprosection import Tensor
from mlprosection.core.backend import Backend, get_default_backend, resolve_backend

from .base import Layer
from .criterion import Criterion
from ..types import Parameter


def _as_array(value: Tensor | Any, backend: Backend):
    if isinstance(value, Tensor):
        return value.data
    return backend.asarray(value)


def _as_index_array(value: Tensor | Any, backend: Backend):
    xp = backend.xp
    if isinstance(value, Tensor):
        return value.data.astype(xp.int64, copy=False)
    return xp.asarray(value, dtype=xp.int64)


def _sigmoid_array(x, xp):
    return 1 / (1 + xp.exp(-x))


def _softmax_array(x, xp):
    if x.ndim == 2:
        x = x - x.max(axis=1, keepdims=True)
        x = xp.exp(x)
        return x / x.sum(axis=1, keepdims=True)
    x = x - x.max()
    exp_x = xp.exp(x)
    return exp_x / exp_x.sum()


def _make_parameter(
    shape: tuple[int, ...],
    *,
    backend: Backend | str | None,
    scale: float,
    name: str,
    zeros: bool = False,
) -> Parameter:
    resolved = resolve_backend(backend) if backend is not None else get_default_backend()
    xp = resolved.xp
    if zeros:
        data = xp.zeros(shape, dtype=resolved.float_dtype)
    else:
        data = (scale * xp.random.randn(*shape)).astype(resolved.float_dtype)
    return Parameter(data, backend=resolved, name=name)


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


class TimeRNN(Layer):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        *,
        stateful: bool = False,
        backend: Backend | str | None = None,
    ) -> None:
        super().__init__(backend)
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
        self.h = None
        self.dh = None
        self.stateful = stateful

    def forward_manual(self, xs: Tensor) -> Tensor:
        xp = xs.backend.xp
        n, time_size, _ = xs.shape
        hidden_size = self.Wh.shape[0]

        if not self.stateful or self.h is None:
            self.h = xp.zeros((n, hidden_size), dtype=xs.dtype)

        hs = xp.empty((n, time_size, hidden_size), dtype=xs.dtype)
        self.layers = []
        h = self.h

        for index in range(time_size):
            x = xs.data[:, index, :]
            h_prev = h
            h = xp.tanh(h_prev @ self.Wh.data + x @ self.Wx.data + self.b.data)
            hs[:, index, :] = h
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


class TimeLSTM(Layer):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        *,
        stateful: bool = False,
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
        self.layers: list[tuple[Any, ...]] = []
        self.h = None
        self.c = None
        self.dh = None
        self.stateful = stateful

    def forward_manual(self, xs: Tensor) -> Tensor:
        xp = xs.backend.xp
        n, time_size, _ = xs.shape
        hidden_size = self.Wh.shape[0]

        if not self.stateful or self.h is None:
            self.h = xp.zeros((n, hidden_size), dtype=xs.dtype)
        if not self.stateful or self.c is None:
            self.c = xp.zeros((n, hidden_size), dtype=xs.dtype)

        hs = xp.empty((n, time_size, hidden_size), dtype=xs.dtype)
        self.layers = []
        h = self.h
        c = self.c

        for index in range(time_size):
            x = xs.data[:, index, :]
            h_prev = h
            c_prev = c
            a = x @ self.Wx.data + h_prev @ self.Wh.data + self.b.data

            f = _sigmoid_array(a[:, :hidden_size], xp)
            g = xp.tanh(a[:, hidden_size : 2 * hidden_size])
            i = _sigmoid_array(a[:, 2 * hidden_size : 3 * hidden_size], xp)
            o = _sigmoid_array(a[:, 3 * hidden_size :], xp)

            c = f * c_prev + g * i
            h = o * xp.tanh(c)
            hs[:, index, :] = h
            self.layers.append((x, h_prev, c_prev, i, f, g, o, c))

        self.h = h
        self.c = c
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
        dc = xp.zeros((n, hidden_size), dtype=dhs.dtype)

        for index in reversed(range(time_size)):
            x, h_prev, c_prev, i, f, g, o, c_next = self.layers[index]
            tanh_c_next = xp.tanh(c_next)
            ds = dc + (dhs.data[:, index, :] + dh) * o * (1 - tanh_c_next**2)
            dc = ds * f
            di = ds * g
            df = ds * c_prev
            do = (dhs.data[:, index, :] + dh) * tanh_c_next
            dg = ds * i

            di *= i * (1 - i)
            df *= f * (1 - f)
            do *= o * (1 - o)
            dg *= 1 - g**2

            da = xp.hstack((df, dg, di, do))
            d_wh += h_prev.T @ da
            d_wx += x.T @ da
            db += da.sum(axis=0)
            dxs[:, index, :] = da @ self.Wx.data.T
            dh = da @ self.Wh.data.T

        self.Wx.grad[...] = d_wx
        self.Wh.grad[...] = d_wh
        self.b.grad[...] = db
        self.dh = Tensor(dh, backend=dhs.backend)
        return Tensor(dxs, backend=dhs.backend)

    def set_state(self, h: Tensor | Any, c: Tensor | Any | None = None) -> None:
        self.h = _as_array(h, self.backend)
        self.c = None if c is None else _as_array(c, self.backend)

    def reset_state(self) -> None:
        self.h = None
        self.c = None


class TimeEmbedding(Layer):
    def __init__(
        self,
        vocab_size: int,
        wordvec_size: int,
        *,
        backend: Backend | str | None = None,
        weight_scale: float = 0.01,
    ) -> None:
        super().__init__(backend)
        self.W = _make_parameter(
            (vocab_size, wordvec_size),
            backend=self._backend,
            scale=weight_scale,
            name="W",
        )
        self.idx = None

    def forward_manual(self, xs: Tensor | Any) -> Tensor:
        idx = _as_index_array(xs, self.backend)
        self.idx = idx
        return Tensor(self.W.data[idx], backend=self.backend)

    def backward_manual(self, dout: Tensor) -> None:
        if self.idx is None:
            raise RuntimeError("forward must be called before backward")

        xp = self.W.backend.xp
        self.W.grad[...] = xp.zeros_like(self.W.data)
        xp.add.at(self.W.grad, self.idx, dout.data)
        return None


class TimeAffine(Layer):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        backend: Backend | str | None = None,
        weight_scale: float | None = None,
        weight: Parameter | None = None,
        transpose_weight: bool = False,
    ) -> None:
        super().__init__(backend or (weight.backend if weight is not None else None))
        scale = weight_scale if weight_scale is not None else 1 / in_features**0.5
        self.W = weight or _make_parameter(
            (in_features, out_features),
            backend=self._backend,
            scale=scale,
            name="W",
        )
        self.b = _make_parameter(
            (out_features,),
            backend=self._backend,
            scale=1.0,
            name="b",
            zeros=True,
        )
        self.transpose_weight = transpose_weight
        self.x = None

    def _weight_data(self):
        return self.W.data.T if self.transpose_weight else self.W.data

    def forward_manual(self, xs: Tensor) -> Tensor:
        n, time_size, _ = xs.shape
        out = xs.data.reshape(n * time_size, -1) @ self._weight_data() + self.b.data
        self.x = xs
        return Tensor(out.reshape(n, time_size, -1), backend=xs.backend)

    def backward_manual(self, dout: Tensor) -> Tensor:
        if self.x is None:
            raise RuntimeError("forward must be called before backward")

        x = self.x
        n, time_size, _ = x.shape
        reshaped_dout = dout.data.reshape(n * time_size, -1)
        reshaped_x = x.data.reshape(n * time_size, -1)

        self.b.grad[...] = reshaped_dout.sum(axis=0)
        d_w = reshaped_x.T @ reshaped_dout
        if self.transpose_weight:
            self.W.grad[...] = d_w.T
        else:
            self.W.grad[...] = d_w

        dx = reshaped_dout @ self._weight_data().T
        return Tensor(dx.reshape(*x.shape), backend=x.backend)


class TimeSoftmaxWithLoss(Criterion):
    def __init__(self, ignore_label: int = -1) -> None:
        super().__init__()
        self.cache = None
        self.ignore_label = ignore_label

    def forward_manual(self, xs: Tensor, ts: Tensor) -> Tensor:
        xp = xs.backend.xp
        n, time_size, vocab_size = xs.shape
        ts_data = ts.data
        if ts.ndim == 3:
            ts_data = ts_data.argmax(axis=2)

        mask = ts_data != self.ignore_label
        scores = xs.data.reshape(n * time_size, vocab_size)
        labels = ts_data.reshape(n * time_size)
        mask = mask.reshape(n * time_size)

        ys = _softmax_array(scores, xp)
        losses = xp.log(ys[xp.arange(n * time_size), labels] + 1e-7)
        losses *= mask
        loss = -xp.sum(losses) / mask.sum()

        self.y = Tensor(ys.reshape(n, time_size, vocab_size), backend=xs.backend)
        self.t = Tensor(labels.reshape(n, time_size), backend=xs.backend)
        self.cache = (labels, ys, mask, (n, time_size, vocab_size), xs.backend)
        return Tensor(loss, backend=xs.backend)

    def backward_manual(self, dout: Tensor | None = None) -> Tensor:
        if self.cache is None:
            raise RuntimeError("forward must be called before backward")

        labels, ys, mask, (n, time_size, vocab_size), backend = self.cache
        xp = backend.xp
        scale = 1.0 if dout is None else float(dout.data)
        dx = ys.copy()
        dx[xp.arange(n * time_size), labels] -= 1
        dx *= scale
        dx /= mask.sum()
        dx *= mask[:, xp.newaxis]
        return Tensor(dx.reshape(n, time_size, vocab_size), backend=backend)


class TimeDropout(Layer):
    def __init__(self, dropout_ratio: float = 0.5) -> None:
        self.dropout_ratio = dropout_ratio
        self.mask = None
        self.train_flg = True

    def forward_manual(self, xs: Tensor) -> Tensor:
        if self.train_flg:
            xp = xs.backend.xp
            mask = xp.random.rand(*xs.shape) > self.dropout_ratio
            scale = 1 / (1.0 - self.dropout_ratio)
            self.mask = mask.astype(xs.dtype) * scale
            return Tensor(xs.data * self.mask, backend=xs.backend)
        return xs

    def backward_manual(self, dout: Tensor) -> Tensor:
        return Tensor(dout.data * self.mask, backend=dout.backend)


class TimeBiLSTM(Layer):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        *,
        stateful: bool = False,
        backend: Backend | str | None = None,
    ) -> None:
        super().__init__(backend)
        self.forward_lstm = TimeLSTM(
            input_size,
            hidden_size,
            stateful=stateful,
            backend=self._backend,
        )
        self.backward_lstm = TimeLSTM(
            input_size,
            hidden_size,
            stateful=stateful,
            backend=self._backend,
        )

    def forward_manual(self, xs: Tensor) -> Tensor:
        out_forward = self.forward_lstm.forward(xs)
        out_backward = self.backward_lstm.forward(xs[:, ::-1])
        out_data = xs.backend.xp.concatenate(
            (out_forward.data, out_backward.data[:, ::-1]),
            axis=2,
        )
        return Tensor(out_data, backend=xs.backend)

    def backward_manual(self, dhs: Tensor) -> Tensor:
        hidden_size = dhs.shape[2] // 2
        dout_forward = dhs[:, :, :hidden_size]
        dout_backward = dhs[:, :, hidden_size:]

        dx_forward = self.forward_lstm.backward(dout_forward)
        dx_backward = self.backward_lstm.backward(dout_backward[:, ::-1])
        return Tensor(dx_forward.data + dx_backward.data[:, ::-1], backend=dhs.backend)


class TimeSigmoidWithLoss(Criterion):
    def __init__(self) -> None:
        super().__init__()
        self.xs_shape = None
        self.layers: list[tuple[Any, Any]] = []
        self._backend = None

    def forward_manual(self, xs: Tensor, ts: Tensor) -> Tensor:
        xp = xs.backend.xp
        n, time_size = xs.shape
        self.xs_shape = xs.shape
        self.layers = []
        self._backend = xs.backend
        loss = 0

        for index in range(time_size):
            y = _sigmoid_array(xs.data[:, index], xp)
            t = ts.data[:, index]
            probs = xp.c_[1 - y, y]
            loss += -xp.log(probs[xp.arange(n), t] + 1e-7).mean()
            self.layers.append((y, t))

        return Tensor(loss / time_size, backend=xs.backend)

    def backward_manual(self, dout: Tensor | None = None) -> Tensor:
        if self.xs_shape is None:
            raise RuntimeError("forward must be called before backward")

        backend = self._backend or get_default_backend()
        xp = backend.xp
        scale = 1.0 if dout is None else float(dout.data)
        n, time_size = self.xs_shape
        dxs = xp.empty(self.xs_shape, dtype=backend.float_dtype)
        for index, (y, t) in enumerate(self.layers):
            dxs[:, index] = (y - t) * scale / n / time_size
        return Tensor(dxs, backend=backend)


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


class TimeGRU(Layer):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        *,
        stateful: bool = False,
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
        self.layers: list[tuple[Any, ...]] = []
        self.h = None
        self.dh = None
        self.stateful = stateful

    def forward_manual(self, xs: Tensor) -> Tensor:
        xp = xs.backend.xp
        n, time_size, _ = xs.shape
        hidden_size = self.Wh.shape[0]
        if not self.stateful or self.h is None:
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


SimpleTimeSoftmaxWithLoss = TimeSoftmaxWithLoss
SimpleTimeAffine = TimeAffine
