"""Recurrent neural network language model architectures."""

from __future__ import annotations

from deepscratch.core import Tensor
from deepscratch.core.backend import Backend
from deepscratch.nn.layers import (
    TimeAffine,
    TimeDropout,
    TimeEmbedding,
    TimeLSTM,
    TimeRNN,
)
from deepscratch.nn.model.base import Model


class Rnnlm(Model):
    def __init__(
        self,
        vocab_size: int = 10000,
        wordvec_size: int = 100,
        hidden_size: int = 100,
        *,
        backend: Backend | str | None = None,
    ) -> None:
        super().__init__(backend)
        self.layers = [
            TimeEmbedding(vocab_size, wordvec_size, backend=self._backend),
            TimeLSTM(
                wordvec_size,
                hidden_size,
                stateful=True,
                backend=self._backend,
            ),
            TimeAffine(hidden_size, vocab_size, backend=self._backend),
        ]
        self.lstm_layer = self.layers[1]

    def predict(self, xs: Tensor, *, cache: bool = True) -> Tensor:
        for layer in self.layers:
            xs = layer.forward(xs, cache=cache)
        return xs

    def forward_manual(self, xs: Tensor, *, cache: bool = True) -> Tensor:
        return self.predict(xs, cache=cache)

    def backward_manual(self, dout: Tensor) -> Tensor:
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout


class TiedRnnlm(Rnnlm):
    """One-layer LSTM RNNLM with input/output embedding weight tying."""

    def __init__(
        self,
        vocab_size: int = 10000,
        wordvec_size: int = 100,
        hidden_size: int = 100,
        *,
        backend: Backend | str | None = None,
    ) -> None:
        Model.__init__(self, backend)
        self.embed = TimeEmbedding(vocab_size, wordvec_size, backend=self._backend)
        self.layers = [
            self.embed,
            TimeLSTM(
                wordvec_size,
                hidden_size,
                stateful=True,
                backend=self._backend,
            ),
            TimeAffine(
                hidden_size,
                vocab_size,
                backend=self._backend,
                weight=self.embed.W,
                transpose_weight=True,
            ),
        ]
        self.lstm_layer = self.layers[1]

    def backward_manual(self, dout: Tensor) -> Tensor:
        tied_grad = None
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
            if layer is self.layers[-1]:
                tied_grad = self.embed.W.grad.copy()
        if tied_grad is not None:
            self.embed.W.grad[...] += tied_grad
        return dout


class VanillaRnnlm(Model):
    """Embedding → vanilla RNN → vocabulary projection language model."""

    def __init__(
        self,
        vocab_size: int = 10000,
        wordvec_size: int = 100,
        hidden_size: int = 100,
        *,
        backend: Backend | str | None = None,
    ) -> None:
        super().__init__(backend)
        self.layers = [
            TimeEmbedding(vocab_size, wordvec_size, backend=self._backend),
            TimeRNN(wordvec_size, hidden_size, stateful=True, backend=self._backend),
            TimeAffine(
                hidden_size,
                vocab_size,
                backend=self._backend,
                weight_scale=1 / hidden_size**0.5,
            ),
        ]
        self.rnn_layer = self.layers[1]

    def predict(self, xs: Tensor, *, cache: bool = True) -> Tensor:
        for layer in self.layers:
            xs = layer.forward(xs, cache=cache)
        return xs

    def forward_manual(self, xs: Tensor, *, cache: bool = True) -> Tensor:
        return self.predict(xs, cache=cache)

    def backward_manual(self, dout: Tensor) -> Tensor:
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout


class BetterRnnlm(Model):
    def __init__(
        self,
        vocab_size: int = 10000,
        wordvec_size: int = 650,
        hidden_size: int = 650,
        dropout_ratio: float = 0.5,
        dropout_rng=None,
        *,
        backend: Backend | str | None = None,
    ) -> None:
        super().__init__(backend)
        self.embed = TimeEmbedding(vocab_size, wordvec_size, backend=self._backend)
        self.layers = [
            self.embed,
            TimeDropout(dropout_ratio, rng=dropout_rng),
            TimeLSTM(
                wordvec_size,
                hidden_size,
                stateful=True,
                backend=self._backend,
            ),
            TimeDropout(dropout_ratio, rng=dropout_rng),
            TimeLSTM(
                hidden_size,
                hidden_size,
                stateful=True,
                backend=self._backend,
            ),
            TimeDropout(dropout_ratio, rng=dropout_rng),
            TimeAffine(
                hidden_size,
                vocab_size,
                backend=self._backend,
                weight=self.embed.W,
                transpose_weight=True,
            ),
        ]
        self.lstm_layers = [self.layers[2], self.layers[4]]
        self.drop_layers = [self.layers[1], self.layers[3], self.layers[5]]

    def predict(self, xs: Tensor, *, cache: bool = True) -> Tensor:
        for layer in self.layers:
            xs = layer.forward(xs, cache=cache)
        return xs

    def forward_manual(self, xs: Tensor, *, cache: bool = True) -> Tensor:
        return self.predict(xs, cache=cache)

    def backward_manual(self, dout: Tensor) -> Tensor:
        tied_grad = None
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
            if layer is self.layers[-1]:
                tied_grad = self.embed.W.grad.copy()

        if tied_grad is not None:
            self.embed.W.grad[...] += tied_grad
        return dout


__all__ = [
    "BetterRnnlm",
    "Rnnlm",
    "TiedRnnlm",
    "VanillaRnnlm",
]
