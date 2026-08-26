"""Frozen selector for the accepted Phase 1 sequence-GEMM implementation."""

from deepscratch.nn.layers import TimeLSTM


class Phase1TimeLSTM(TimeLSTM):
    """Production TimeLSTM with later fused CUDA phases disabled."""

    _fused_cuda = False


def replace_better_rnnlm_lstms(model) -> None:
    for index in (2, 4):
        old = model.layers[index]
        layer = Phase1TimeLSTM(
            old.Wx.shape[0], old.Wh.shape[0], stateful=True, backend=old.backend
        )
        layer.Wx.data[...] = old.Wx.data
        layer.Wh.data[...] = old.Wh.data
        layer.b.data[...] = old.b.data
        model.layers[index] = layer
    model.lstm_layers = [model.layers[2], model.layers[4]]
