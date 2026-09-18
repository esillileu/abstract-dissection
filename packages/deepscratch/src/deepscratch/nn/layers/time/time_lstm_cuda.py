from __future__ import annotations

from deepscratch.core import Tensor

from ...kernels.time_lstm import launch_backward as _launch_lstm_backward
from ...kernels.time_lstm import launch_forward as _launch_lstm_forward


def forward_cuda_float32(layer, xs: Tensor, *, cache: bool) -> Tensor:
    xp = xs.backend.xp
    n, time_size, input_size = xs.shape
    hidden_size = layer.Wh.shape[0]
    if not layer.stateful or layer.h is None or layer.h.shape[0] != n:
        layer.h = xp.zeros((n, hidden_size), dtype=xs.dtype)
    if not layer.stateful or layer.c is None or layer.c.shape[0] != n:
        layer.c = xp.zeros((n, hidden_size), dtype=xs.dtype)

    x_flat = xp.ascontiguousarray(xs.data.reshape(n * time_size, input_size))
    with layer.backend.range("TimeLSTM/forward_input_gemm"):
        xproj = (x_flat @ layer.Wx.data).reshape(n, time_size, 4 * hidden_size)
    hs = xp.empty((n, time_size, hidden_size), dtype=xs.dtype)
    layer.layers = []
    if cache:
        hpseq = xp.empty_like(hs)
        cpseq = xp.empty_like(hs)
        gates = xp.empty((n, time_size, 4 * hidden_size), dtype=xs.dtype)
        cells = xp.empty_like(hs)
    else:
        hpseq = cpseq = gates = cells = hs

    h_work = xp.empty((2, n, hidden_size), dtype=xs.dtype)
    c_work = xp.empty_like(h_work)
    h, c = layer.h, layer.c
    with layer.backend.range("TimeLSTM/forward_recurrent_loop"):
        for index in range(time_size):
            h_prev, c_prev = h, c
            hproj = h_prev @ layer.Wh.data
            h, c = h_work[index & 1], c_work[index & 1]
            _launch_lstm_forward(
                xproj,
                hproj,
                layer.b.data,
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
        layer.layers.append((x_flat, hpseq, cpseq, gates, cells))
    layer.h, layer.c = h, c
    return Tensor(hs, backend=xs.backend)


def backward_cuda_float32(layer, dhs: Tensor) -> Tensor:
    if not layer.layers:
        raise RuntimeError("forward must be called before backward")
    xp = dhs.backend.xp
    # Raw kernels index ``dhs`` as a packed (N, T, H) array.  Callers such
    # as PeekyDecoder can pass a sliced view whose last dimension is
    # contiguous but whose batch/time strides still belong to (N, T, 2H).
    dhs_data = xp.ascontiguousarray(dhs.data)
    n, time_size, hidden_size = dhs.shape
    input_size = layer.Wx.shape[0]
    x_flat, hpseq, cpseq, gates, cells = layer.layers[0]
    daseq = xp.empty((n, time_size, 4 * hidden_size), dtype=dhs.dtype)
    dh = xp.zeros((n, hidden_size), dtype=dhs.dtype)
    dc = xp.zeros_like(dh)
    dc_work = xp.empty((2, n, hidden_size), dtype=dhs.dtype)
    with layer.backend.range("TimeLSTM/backward_recurrent_loop"):
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
            dh = daseq[:, index, :] @ layer.Wh.data.T

    da_flat = daseq.reshape(n * time_size, 4 * hidden_size)
    hp_flat = hpseq.reshape(n * time_size, hidden_size)
    with layer.backend.range("TimeLSTM/backward_dWx_gemm"):
        layer.Wx.grad[...] = x_flat.T @ da_flat
    with layer.backend.range("TimeLSTM/backward_dWh_gemm"):
        layer.Wh.grad[...] = hp_flat.T @ da_flat
    layer.b.grad[...] = da_flat.sum(axis=0)
    with layer.backend.range("TimeLSTM/backward_dX_gemm"):
        dxs = (da_flat @ layer.Wx.data.T).reshape(n, time_size, input_size)
    layer.dh = Tensor(dh, backend=dhs.backend)
    return Tensor(dxs, backend=dhs.backend)
