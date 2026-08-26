from __future__ import annotations

from typing import Any

from deepscratch.core import Tensor
from deepscratch.core.backend import Backend, get_default_backend, resolve_backend

from ..kernels.time_lstm import launch_backward as _launch_lstm_backward
from ..kernels.time_lstm import launch_forward as _launch_lstm_forward
from ..types import Parameter
from .base import Layer
from .embeding import Embedding
from .linear import Affine
from .regulizer import BatchNormalization, Dropout


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


def _make_parameter(
    shape: tuple[int, ...],
    *,
    backend: Backend | str | None,
    scale: float,
    name: str,
    zeros: bool = False,
) -> Parameter:
    resolved = (
        resolve_backend(backend) if backend is not None else get_default_backend()
    )
    xp = resolved.xp
    if zeros:
        data = xp.zeros(shape, dtype=resolved.float_dtype)
    else:
        data = (scale * resolved.random_stream("model_init").randn(*shape)).astype(
            resolved.float_dtype
        )
    return Parameter(data, backend=resolved, name=name)


class TimeLayer(Layer):
    """Base contract for layers whose leading shape is ``(batch, time, ...)``."""

    time_axis = 1

    def reset_state(self) -> None:
        """Reset optional state carried between truncated-BPTT batches."""

    def detach_state(self) -> None:
        """Keep values but clear backward-time caches at a BPTT boundary."""


class TimeDistributed(TimeLayer):
    """Apply one ordinary layer to every timestep with shared parameters.

    The wrapped layer is called once on a flattened ``batch * time`` batch, so
    its parameter gradients naturally aggregate across all timesteps.
    """

    def __init__(self, layer: Layer) -> None:
        super().__init__(layer.backend)
        self.layer = layer
        self.input_shape: tuple[int, ...] | None = None
        self.output_shape: tuple[int, ...] | None = None

    def forward_manual(self, xs: Tensor, *, cache: bool = True) -> Tensor:
        if xs.ndim < 2:
            raise ValueError("TimeDistributed expects (batch, time, ...) input")
        self.input_shape = xs.shape
        flat = Tensor(
            xs.data.reshape(xs.shape[0] * xs.shape[1], *xs.shape[2:]),
            backend=xs.backend,
        )
        output = self.layer.forward(flat)
        self.output_shape = output.shape[1:]
        result = Tensor(
            output.data.reshape(xs.shape[0], xs.shape[1], *self.output_shape),
            backend=output.backend,
        )
        if not cache:
            self.input_shape = None
            self.output_shape = None
            for name in ("x", "idx", "mask", "cache"):
                if hasattr(self.layer, name):
                    setattr(self.layer, name, None)
        return result

    def backward_manual(self, dout: Tensor):
        if self.input_shape is None or self.output_shape is None:
            raise RuntimeError("forward must be called before backward")
        flat = Tensor(
            dout.data.reshape(
                self.input_shape[0] * self.input_shape[1], *self.output_shape
            ),
            backend=dout.backend,
        )
        dx = self.layer.backward(flat)
        if dx is None:
            return None
        return Tensor(dx.data.reshape(*self.input_shape), backend=dx.backend)


class RecurrentTimeLayer(TimeLayer):
    """Common state lifecycle for recurrent time layers."""

    def __init__(
        self, *, stateful: bool = False, backend: Backend | str | None = None
    ) -> None:
        super().__init__(backend)
        self.stateful = stateful

    def detach_state(self) -> None:
        layers = getattr(self, "layers", None)
        if isinstance(layers, list):
            layers.clear()


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
        xp = xs.backend.xp
        n, time_size, input_size = xs.shape
        hidden_size = self.Wh.shape[0]
        if not self.stateful or self.h is None or self.h.shape[0] != n:
            self.h = xp.zeros((n, hidden_size), dtype=xs.dtype)
        if not self.stateful or self.c is None or self.c.shape[0] != n:
            self.c = xp.zeros((n, hidden_size), dtype=xs.dtype)

        x_flat = xp.ascontiguousarray(xs.data.reshape(n * time_size, input_size))
        with self.backend.range("TimeLSTM/forward_input_gemm"):
            xproj = (x_flat @ self.Wx.data).reshape(n, time_size, 4 * hidden_size)
        hs = xp.empty((n, time_size, hidden_size), dtype=xs.dtype)
        self.layers = []
        if cache:
            hpseq = xp.empty_like(hs)
            cpseq = xp.empty_like(hs)
            gates = xp.empty((n, time_size, 4 * hidden_size), dtype=xs.dtype)
            cells = xp.empty_like(hs)
        else:
            hpseq = cpseq = gates = cells = hs

        h_work = xp.empty((2, n, hidden_size), dtype=xs.dtype)
        c_work = xp.empty_like(h_work)
        h, c = self.h, self.c
        with self.backend.range("TimeLSTM/forward_recurrent_loop"):
            for index in range(time_size):
                h_prev, c_prev = h, c
                hproj = h_prev @ self.Wh.data
                h, c = h_work[index & 1], c_work[index & 1]
                _launch_lstm_forward(
                    xproj,
                    hproj,
                    self.b.data,
                    h_prev,
                    c_prev,
                    h,
                    c,
                    hs,
                    hpseq,
                    cpseq,
                    gates,
                    cells,
                    timestep=index,
                    cache=cache,
                )
        if cache:
            self.layers.append((x_flat, hpseq, cpseq, gates, cells))
        self.h, self.c = h, c
        return Tensor(hs, backend=xs.backend)

    def _backward_cuda_float32(self, dhs: Tensor) -> Tensor:
        if not self.layers:
            raise RuntimeError("forward must be called before backward")
        xp = dhs.backend.xp
        # Raw kernels index ``dhs`` as a packed (N, T, H) array.  Callers such
        # as PeekyDecoder can pass a sliced view whose last dimension is
        # contiguous but whose batch/time strides still belong to (N, T, 2H).
        dhs_data = xp.ascontiguousarray(dhs.data)
        n, time_size, hidden_size = dhs.shape
        input_size = self.Wx.shape[0]
        x_flat, hpseq, cpseq, gates, cells = self.layers[0]
        daseq = xp.empty((n, time_size, 4 * hidden_size), dtype=dhs.dtype)
        dh = xp.zeros((n, hidden_size), dtype=dhs.dtype)
        dc = xp.zeros_like(dh)
        dc_work = xp.empty((2, n, hidden_size), dtype=dhs.dtype)
        with self.backend.range("TimeLSTM/backward_recurrent_loop"):
            for index in reversed(range(time_size)):
                dc_prev = dc_work[index & 1]
                _launch_lstm_backward(
                    dhs_data,
                    dh,
                    dc,
                    cpseq,
                    gates,
                    cells,
                    daseq,
                    dc_prev,
                    timestep=index,
                )
                dc = dc_prev
                dh = daseq[:, index, :] @ self.Wh.data.T

        da_flat = daseq.reshape(n * time_size, 4 * hidden_size)
        hp_flat = hpseq.reshape(n * time_size, hidden_size)
        with self.backend.range("TimeLSTM/backward_dWx_gemm"):
            self.Wx.grad[...] = x_flat.T @ da_flat
        with self.backend.range("TimeLSTM/backward_dWh_gemm"):
            self.Wh.grad[...] = hp_flat.T @ da_flat
        self.b.grad[...] = da_flat.sum(axis=0)
        with self.backend.range("TimeLSTM/backward_dX_gemm"):
            dxs = (da_flat @ self.Wx.data.T).reshape(n, time_size, input_size)
        self.dh = Tensor(dh, backend=dhs.backend)
        return Tensor(dxs, backend=dhs.backend)

    def set_state(self, h: Tensor | Any, c: Tensor | Any | None = None) -> None:
        self.h = _as_array(h, self.backend)
        self.c = None if c is None else _as_array(c, self.backend)

    def reset_state(self) -> None:
        self.h = None
        self.c = None


class TimeEmbedding(TimeDistributed):
    def __init__(
        self,
        vocab_size: int,
        wordvec_size: int,
        *,
        backend: Backend | str | None = None,
        weight_scale: float = 0.01,
    ) -> None:
        resolved = (
            resolve_backend(backend) if backend is not None else get_default_backend()
        )
        layer = Embedding(vocab_size, wordvec_size, backend=resolved)
        super().__init__(layer)
        self.weight_scale = weight_scale
        if weight_scale != 0.01:
            self.W.data[...] *= weight_scale / 0.01

    @property
    def W(self) -> Parameter:
        return self.layer.W


class TimeAffine(TimeDistributed):
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
        resolved = backend or (weight.backend if weight is not None else None)
        layer = Affine(
            in_features,
            out_features,
            backend=resolved,
            weight=weight,
            transpose_weight=transpose_weight,
        )
        super().__init__(layer)
        if weight_scale is not None and weight is None:
            self.W.data[...] *= weight_scale / 0.01

    @property
    def W(self) -> Parameter:
        return self.layer.W

    @property
    def b(self) -> Parameter:
        return self.layer.b


class TimeDropout(TimeDistributed):
    def __init__(self, dropout_ratio: float = 0.5, *, rng=None) -> None:
        super().__init__(Dropout(dropout_ratio, inverted=True, rng=rng))
        self.dropout_ratio = dropout_ratio


class TimeBatchNormalization(TimeDistributed):
    """Batch-normalize over the flattened batch-and-time population."""

    def __init__(self, **kwargs) -> None:
        super().__init__(BatchNormalization(**kwargs))


class TimeBiLSTM(TimeLayer):
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

    def reset_state(self) -> None:
        self.forward_lstm.reset_state()
        self.backward_lstm.reset_state()

    def detach_state(self) -> None:
        self.forward_lstm.detach_state()
        self.backward_lstm.detach_state()


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


class TimeAttention(TimeLayer):
    """Dot-product attention over encoder and decoder time states."""

    def __init__(self, *, backend: Backend | str | None = None) -> None:
        super().__init__(backend)
        self.cache: tuple[Tensor, Tensor] | None = None
        self.weights = None

    def forward_manual(
        self,
        enc_hs: Tensor,
        dec_hs: Tensor,
        *,
        cache: bool = True,
    ) -> Tensor:
        xp = enc_hs.backend.xp
        scores = xp.sum(enc_hs.data[:, None, :, :] * dec_hs.data[:, :, None, :], axis=3)
        scores -= scores.max(axis=2, keepdims=True)
        weights = xp.exp(scores)
        weights /= weights.sum(axis=2, keepdims=True)
        self.weights = weights if cache else None
        self.cache = (enc_hs, dec_hs) if cache else None
        context = xp.sum(weights[:, :, :, None] * enc_hs.data[:, None, :, :], axis=2)
        return Tensor(context, backend=enc_hs.backend)

    def backward_manual(self, dout: Tensor) -> tuple[Tensor, Tensor]:
        if self.cache is None or self.weights is None:
            raise RuntimeError("forward must be called before backward")
        enc_hs, dec_hs = self.cache
        xp = dout.backend.xp
        dweights = xp.sum(dout.data[:, :, None, :] * enc_hs.data[:, None, :, :], axis=3)
        denc = xp.sum(self.weights[:, :, :, None] * dout.data[:, :, None, :], axis=1)
        dscores = self.weights * (
            dweights - xp.sum(dweights * self.weights, axis=2, keepdims=True)
        )
        denc += xp.sum(dscores[:, :, :, None] * dec_hs.data[:, :, None, :], axis=1)
        ddec = xp.sum(dscores[:, :, :, None] * enc_hs.data[:, None, :, :], axis=2)
        return Tensor(denc, backend=dout.backend), Tensor(ddec, backend=dout.backend)


SimpleTimeAffine = TimeAffine
