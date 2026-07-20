from __future__ import annotations

from mlprosection import Tensor
from mlprosection.core.backend import Backend
from mlprosection.nn.layers import (
    Layer,
    TimeAffine,
    TimeDropout,
    TimeEmbedding,
    TimeLSTM,
    TimeRNN,
    TimeSoftmaxWithLoss,
)


class Rnnlm(Layer):
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
        self.loss_layer = TimeSoftmaxWithLoss()
        self.lstm_layer = self.layers[1]
        self._uses_internal_loss = False

    def predict(self, xs: Tensor) -> Tensor:
        for layer in self.layers:
            xs = layer.forward(xs)
        return xs

    def forward_manual(self, xs: Tensor, ts: Tensor | None = None) -> Tensor:
        score = self.predict(xs)
        self._uses_internal_loss = ts is not None
        if ts is None:
            return score
        return self.loss_layer.forward(score, ts)

    def backward_manual(self, dout: Tensor | int | float | None = None) -> Tensor:
        if self._uses_internal_loss:
            dout = self.loss_layer.backward(None if dout is None else Tensor(dout))
        elif not isinstance(dout, Tensor):
            raise TypeError("dout must be a Tensor when no internal loss was used")

        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def reset_state(self) -> None:
        self.lstm_layer.reset_state()


class VanillaRnnlm(Layer):
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
            TimeAffine(hidden_size, vocab_size, backend=self._backend),
        ]
        self.rnn_layer = self.layers[1]
        self.loss_layer = TimeSoftmaxWithLoss()
        self._uses_internal_loss = False

    def predict(self, xs: Tensor) -> Tensor:
        for layer in self.layers:
            xs = layer.forward(xs)
        return xs

    def forward_manual(self, xs: Tensor, ts: Tensor | None = None) -> Tensor:
        score = self.predict(xs)
        self._uses_internal_loss = ts is not None
        return score if ts is None else self.loss_layer.forward(score, ts)

    def backward_manual(self, dout: Tensor | int | float | None = None) -> Tensor:
        if self._uses_internal_loss:
            dout = self.loss_layer.backward(None if dout is None else Tensor(dout))
        elif not isinstance(dout, Tensor):
            raise TypeError("dout must be a Tensor when no internal loss was used")
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def reset_state(self) -> None:
        self.rnn_layer.reset_state()


class BetterRnnlm(Layer):
    def __init__(
        self,
        vocab_size: int = 10000,
        wordvec_size: int = 650,
        hidden_size: int = 650,
        dropout_ratio: float = 0.5,
        *,
        backend: Backend | str | None = None,
    ) -> None:
        super().__init__(backend)
        self.embed = TimeEmbedding(vocab_size, wordvec_size, backend=self._backend)
        self.layers = [
            self.embed,
            TimeDropout(dropout_ratio),
            TimeLSTM(
                wordvec_size,
                hidden_size,
                stateful=True,
                backend=self._backend,
            ),
            TimeDropout(dropout_ratio),
            TimeLSTM(
                hidden_size,
                hidden_size,
                stateful=True,
                backend=self._backend,
            ),
            TimeDropout(dropout_ratio),
            TimeAffine(
                hidden_size,
                vocab_size,
                backend=self._backend,
                weight=self.embed.W,
                transpose_weight=True,
            ),
        ]
        self.loss_layer = TimeSoftmaxWithLoss()
        self.lstm_layers = [self.layers[2], self.layers[4]]
        self.drop_layers = [self.layers[1], self.layers[3], self.layers[5]]
        self._uses_internal_loss = False

    def predict(self, xs: Tensor) -> Tensor:
        for layer in self.layers:
            xs = layer.forward(xs)
        return xs

    def forward_manual(
        self,
        xs: Tensor,
        ts: Tensor | None = None,
    ) -> Tensor:
        score = self.predict(xs)
        self._uses_internal_loss = ts is not None
        if ts is None:
            return score
        return self.loss_layer.forward(score, ts)

    def backward_manual(self, dout: Tensor | int | float | None = None) -> Tensor:
        if self._uses_internal_loss:
            dout = self.loss_layer.backward(None if dout is None else Tensor(dout))
        elif not isinstance(dout, Tensor):
            raise TypeError("dout must be a Tensor when no internal loss was used")

        tied_grad = None
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
            if layer is self.layers[-1]:
                tied_grad = self.embed.W.grad.copy()

        if tied_grad is not None:
            self.embed.W.grad[...] += tied_grad
        return dout

    def reset_state(self) -> None:
        for layer in self.lstm_layers:
            layer.reset_state()


class Encoder(Layer):
    def __init__(
        self,
        vocab_size: int,
        wordvec_size: int,
        hidden_size: int,
        *,
        backend: Backend | str | None = None,
    ) -> None:
        super().__init__(backend)
        self.embed = TimeEmbedding(vocab_size, wordvec_size, backend=self._backend)
        self.lstm = TimeLSTM(
            wordvec_size,
            hidden_size,
            stateful=False,
            backend=self._backend,
        )
        self.hs: Tensor | None = None

    def forward_manual(self, xs: Tensor) -> Tensor:
        xs = self.embed.forward(xs)
        hs = self.lstm.forward(xs)
        self.hs = hs
        return hs[:, -1, :]

    def backward_manual(self, dh: Tensor) -> None:
        if self.hs is None:
            raise RuntimeError("forward must be called before backward")

        xp = dh.backend.xp
        dhs = xp.zeros_like(self.hs.data)
        dhs[:, -1, :] = dh.data
        dout = self.lstm.backward(Tensor(dhs, backend=dh.backend))
        self.embed.backward(dout)
        return None


class Decoder(Layer):
    def __init__(
        self,
        vocab_size: int,
        wordvec_size: int,
        hidden_size: int,
        *,
        backend: Backend | str | None = None,
    ) -> None:
        super().__init__(backend)
        self.embed = TimeEmbedding(vocab_size, wordvec_size, backend=self._backend)
        self.lstm = TimeLSTM(
            wordvec_size,
            hidden_size,
            stateful=True,
            backend=self._backend,
        )
        self.affine = TimeAffine(hidden_size, vocab_size, backend=self._backend)

    def forward_manual(self, xs: Tensor, h: Tensor) -> Tensor:
        self.lstm.set_state(h)
        out = self.embed.forward(xs)
        out = self.lstm.forward(out)
        return self.affine.forward(out)

    def backward_manual(self, dscore: Tensor) -> Tensor:
        dout = self.affine.backward(dscore)
        dout = self.lstm.backward(dout)
        self.embed.backward(dout)
        return self.lstm.dh

    def generate(self, h: Tensor, start_id: int, sample_size: int) -> list[int]:
        backend = h.backend
        xp = backend.xp
        sampled = []
        sample_id = start_id
        self.lstm.set_state(h)

        for _ in range(sample_size):
            x = Tensor(xp.array(sample_id).reshape((1, 1)), backend=backend)
            out = self.embed.forward(x)
            out = self.lstm.forward(out)
            score = self.affine.forward(out)
            sample_id = int(score.data.flatten().argmax())
            sampled.append(sample_id)

        return sampled


class Seq2seq(Layer):
    def __init__(
        self,
        vocab_size: int,
        wordvec_size: int,
        hidden_size: int,
        *,
        backend: Backend | str | None = None,
    ) -> None:
        super().__init__(backend)
        self.encoder = Encoder(
            vocab_size,
            wordvec_size,
            hidden_size,
            backend=self._backend,
        )
        self.decoder = Decoder(
            vocab_size,
            wordvec_size,
            hidden_size,
            backend=self._backend,
        )
        self.softmax = TimeSoftmaxWithLoss()

    def forward_manual(self, xs: Tensor, ts: Tensor) -> Tensor:
        decoder_xs = ts[:, :-1]
        decoder_ts = ts[:, 1:]

        h = self.encoder.forward(xs)
        score = self.decoder.forward(decoder_xs, h)
        return self.softmax.forward(score, decoder_ts)

    def backward_manual(self, dout: Tensor | int | float | None = None) -> None:
        if dout is None:
            dout = None
        elif not isinstance(dout, Tensor):
            dout = Tensor(dout)

        dout = self.softmax.backward(dout)
        dh = self.decoder.backward(dout)
        self.encoder.backward(dh)
        return None

    def generate(self, xs: Tensor, start_id: int, sample_size: int) -> list[int]:
        h = self.encoder.forward(xs)
        return self.decoder.generate(h, start_id, sample_size)


class PeekyDecoder(Decoder):
    """Decoder that concatenates encoder state to each input and output state."""

    def __init__(self, vocab_size: int, wordvec_size: int, hidden_size: int, *, backend: Backend | str | None = None) -> None:
        Layer.__init__(self, backend)
        self.hidden_size = hidden_size
        self.embed = TimeEmbedding(vocab_size, wordvec_size, backend=self._backend)
        self.lstm = TimeLSTM(wordvec_size + hidden_size, hidden_size, stateful=True, backend=self._backend)
        self.affine = TimeAffine(hidden_size * 2, vocab_size, backend=self._backend)
        self.peeky_h = None

    def forward_manual(self, xs: Tensor, h: Tensor) -> Tensor:
        xp = h.backend.xp
        self.lstm.set_state(h)
        out = self.embed.forward(xs)
        self.peeky_h = xp.repeat(h.data[:, None, :], xs.shape[1], axis=1)
        out = Tensor(xp.concatenate((self.peeky_h, out.data), axis=2), backend=h.backend)
        out = self.lstm.forward(out)
        out = Tensor(xp.concatenate((self.peeky_h, out.data), axis=2), backend=h.backend)
        return self.affine.forward(out)

    def backward_manual(self, dscore: Tensor) -> Tensor:
        if self.peeky_h is None:
            raise RuntimeError("forward must be called before backward")
        h = self.hidden_size
        dout = self.affine.backward(dscore)
        d_lstm_out, d_peeky_out = dout.data[:, :, h:], dout.data[:, :, :h]
        dout = self.lstm.backward(Tensor(d_lstm_out, backend=dscore.backend))
        d_embed, d_peeky_in = dout.data[:, :, h:], dout.data[:, :, :h]
        self.embed.backward(Tensor(d_embed, backend=dscore.backend))
        return Tensor(self.lstm.dh.data + (d_peeky_out + d_peeky_in).sum(axis=1), backend=dscore.backend)

    def generate(self, h: Tensor, start_id: int, sample_size: int) -> list[int]:
        xp = h.backend.xp
        self.lstm.set_state(h)
        sample_id = start_id
        sampled = []
        peeky_h = h.data[:, None, :]
        for _ in range(sample_size):
            x = Tensor(xp.asarray([[sample_id]], dtype=xp.int64), backend=h.backend)
            out = self.embed.forward(x)
            out = Tensor(xp.concatenate((peeky_h, out.data), axis=2), backend=h.backend)
            out = self.lstm.forward(out)
            score = self.affine.forward(Tensor(xp.concatenate((peeky_h, out.data), axis=2), backend=h.backend))
            sample_id = int(score.data.reshape(-1).argmax())
            sampled.append(sample_id)
        return sampled


class PeekySeq2seq(Seq2seq):
    def __init__(self, vocab_size: int, wordvec_size: int, hidden_size: int, *, backend: Backend | str | None = None) -> None:
        Layer.__init__(self, backend)
        self.encoder = Encoder(vocab_size, wordvec_size, hidden_size, backend=self._backend)
        self.decoder = PeekyDecoder(vocab_size, wordvec_size, hidden_size, backend=self._backend)
        self.softmax = TimeSoftmaxWithLoss()


class TimeAttention(Layer):
    """Dot-product attention over encoder time states."""

    def __init__(self, *, backend: Backend | str | None = None) -> None:
        super().__init__(backend)
        self.cache = None
        self.weights = None

    def forward_manual(self, enc_hs: Tensor, dec_hs: Tensor) -> Tensor:
        xp = enc_hs.backend.xp
        scores = xp.sum(enc_hs.data[:, None, :, :] * dec_hs.data[:, :, None, :], axis=3)
        scores -= scores.max(axis=2, keepdims=True)
        weights = xp.exp(scores)
        weights /= weights.sum(axis=2, keepdims=True)
        self.weights = weights
        self.cache = (enc_hs, dec_hs)
        return Tensor(xp.sum(weights[:, :, :, None] * enc_hs.data[:, None, :, :], axis=2), backend=enc_hs.backend)

    def backward_manual(self, dout: Tensor) -> tuple[Tensor, Tensor]:
        if self.cache is None or self.weights is None:
            raise RuntimeError("forward must be called before backward")
        enc_hs, dec_hs = self.cache
        xp = dout.backend.xp
        dweights = xp.sum(dout.data[:, :, None, :] * enc_hs.data[:, None, :, :], axis=3)
        denc = xp.sum(self.weights[:, :, :, None] * dout.data[:, :, None, :], axis=1)
        dscores = self.weights * (dweights - xp.sum(dweights * self.weights, axis=2, keepdims=True))
        denc += xp.sum(dscores[:, :, :, None] * dec_hs.data[:, :, None, :], axis=1)
        ddec = xp.sum(dscores[:, :, :, None] * enc_hs.data[:, None, :, :], axis=2)
        return Tensor(denc, backend=dout.backend), Tensor(ddec, backend=dout.backend)


class AttentionEncoder(Encoder):
    def forward_manual(self, xs: Tensor) -> Tensor:
        return self.lstm.forward(self.embed.forward(xs))

    def backward_manual(self, dhs: Tensor) -> None:
        self.embed.backward(self.lstm.backward(dhs))
        return None


class AttentionDecoder(Layer):
    def __init__(self, vocab_size: int, wordvec_size: int, hidden_size: int, *, backend: Backend | str | None = None) -> None:
        super().__init__(backend)
        self.embed = TimeEmbedding(vocab_size, wordvec_size, backend=self._backend)
        self.lstm = TimeLSTM(wordvec_size, hidden_size, stateful=True, backend=self._backend)
        self.attention = TimeAttention(backend=self._backend)
        self.affine = TimeAffine(hidden_size * 2, vocab_size, backend=self._backend)

    def forward_manual(self, xs: Tensor, enc_hs: Tensor) -> Tensor:
        self.lstm.set_state(enc_hs[:, -1, :])
        dec_hs = self.lstm.forward(self.embed.forward(xs))
        context = self.attention.forward(enc_hs, dec_hs)
        xp = xs.backend.xp
        return self.affine.forward(Tensor(xp.concatenate((context.data, dec_hs.data), axis=2), backend=xs.backend))

    def backward_manual(self, dscore: Tensor) -> Tensor:
        dout = self.affine.backward(dscore)
        hidden = dout.shape[2] // 2
        denc, ddec = self.attention.backward(Tensor(dout.data[:, :, :hidden], backend=dscore.backend))
        dx = self.lstm.backward(Tensor(dout.data[:, :, hidden:] + ddec.data, backend=dscore.backend))
        denc.data[:, -1, :] += self.lstm.dh.data
        self.embed.backward(dx)
        return denc

    def generate(self, enc_hs: Tensor, start_id: int, sample_size: int) -> list[int]:
        xp = enc_hs.backend.xp
        self.lstm.set_state(enc_hs[:, -1, :])
        sample_id, sampled = start_id, []
        for _ in range(sample_size):
            out = self.embed.forward(Tensor(xp.asarray([[sample_id]], dtype=xp.int64), backend=enc_hs.backend))
            dec_hs = self.lstm.forward(out)
            context = self.attention.forward(enc_hs, dec_hs)
            score = self.affine.forward(Tensor(xp.concatenate((context.data, dec_hs.data), axis=2), backend=enc_hs.backend))
            sample_id = int(score.data.reshape(-1).argmax())
            sampled.append(sample_id)
        return sampled


class AttentionSeq2seq(Seq2seq):
    def __init__(self, vocab_size: int, wordvec_size: int, hidden_size: int, *, backend: Backend | str | None = None) -> None:
        Layer.__init__(self, backend)
        self.encoder = AttentionEncoder(vocab_size, wordvec_size, hidden_size, backend=self._backend)
        self.decoder = AttentionDecoder(vocab_size, wordvec_size, hidden_size, backend=self._backend)
        self.softmax = TimeSoftmaxWithLoss()

    def generate(self, xs: Tensor, start_id: int, sample_size: int) -> list[int]:
        hs = self.encoder.forward(xs)
        return self.decoder.generate(hs, start_id, sample_size)


__all__ = [
    "VanillaRnnlm",
    "Rnnlm",
    "BetterRnnlm",
    "Encoder",
    "Decoder",
    "Seq2seq",
    "PeekySeq2seq",
    "AttentionSeq2seq",
    "TimeAttention",
]
