from __future__ import annotations

from deepscratch.core import Tensor
from deepscratch.core.backend import Backend
from deepscratch.nn.layers import (
    Layer,
    TimeAffine,
    TimeAttention,
    TimeDropout,
    TimeEmbedding,
    TimeLSTM,
    TimeRNN,
)
from deepscratch.nn.model.base import GenerativeModel, Model


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

    def forward_manual(self, xs: Tensor, *, cache: bool = True) -> Tensor:
        xs = self.embed.forward(xs, cache=cache)
        hs = self.lstm.forward(xs, cache=cache)
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
        self.affine = TimeAffine(
            hidden_size,
            vocab_size,
            backend=self._backend,
            weight_scale=1 / hidden_size**0.5,
        )

    def forward_manual(self, xs: Tensor, h: Tensor, *, cache: bool = True) -> Tensor:
        self.lstm.set_state(h)
        out = self.embed.forward(xs, cache=cache)
        out = self.lstm.forward(out, cache=cache)
        return self.affine.forward(out, cache=cache)

    def backward_manual(self, dscore: Tensor) -> Tensor:
        dout = self.affine.backward(dscore)
        dout = self.lstm.backward(dout)
        self.embed.backward(dout)
        return self.lstm.dh

    def generate(self, h: Tensor, start_id: int, sample_size: int) -> list[int]:
        return _host_sampled_ids(
            h.backend, self.generate_device(h, start_id, sample_size)
        )

    def generate_device(self, h: Tensor, start_id: int, sample_size: int):
        backend = h.backend
        xp = backend.xp
        sampled = []
        batch_size = h.shape[0]
        sample_id = xp.full((batch_size,), start_id, dtype=xp.int64)
        self.lstm.set_state(h)

        for _ in range(sample_size):
            x = Tensor(sample_id.reshape((batch_size, 1)), backend=backend)
            out = self.embed.forward(x)
            out = self.lstm.forward(out)
            score = self.affine.forward(out)
            sample_id = score.data[:, -1, :].argmax(axis=1)
            sampled.append(sample_id)

        return _stack_sampled_ids_device(backend, sampled)


class Seq2seq(GenerativeModel):
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

    def forward_manual(
        self, xs: Tensor, decoder_xs: Tensor, *, cache: bool = True
    ) -> Tensor:
        h = self.encoder.forward(xs, cache=cache)
        score = self.decoder.forward(decoder_xs, h, cache=cache)
        return score

    def backward_manual(self, dout: Tensor) -> None:
        dh = self.decoder.backward(dout)
        self.encoder.backward(dh)

    def generate(self, xs: Tensor, start_id: int, sample_size: int) -> list[int]:
        return _host_sampled_ids(
            xs.backend, self.generate_device(xs, start_id, sample_size)
        )

    def generate_device(self, xs: Tensor, start_id: int, sample_size: int):
        h = self.encoder.forward(xs)
        return self.decoder.generate_device(h, start_id, sample_size)


class PeekyDecoder(Decoder):
    """Decoder that concatenates encoder state to each input and output state."""

    def __init__(
        self,
        vocab_size: int,
        wordvec_size: int,
        hidden_size: int,
        *,
        backend: Backend | str | None = None,
    ) -> None:
        Layer.__init__(self, backend)
        self.hidden_size = hidden_size
        self.embed = TimeEmbedding(vocab_size, wordvec_size, backend=self._backend)
        self.lstm = TimeLSTM(
            wordvec_size + hidden_size,
            hidden_size,
            stateful=True,
            backend=self._backend,
        )
        self.affine = TimeAffine(
            hidden_size * 2,
            vocab_size,
            backend=self._backend,
            weight_scale=1 / (hidden_size * 2) ** 0.5,
        )
        self.peeky_h = None

    def forward_manual(self, xs: Tensor, h: Tensor, *, cache: bool = True) -> Tensor:
        xp = h.backend.xp
        self.lstm.set_state(h)
        out = self.embed.forward(xs, cache=cache)
        self.peeky_h = xp.repeat(h.data[:, None, :], xs.shape[1], axis=1)
        out = Tensor(
            xp.concatenate((self.peeky_h, out.data), axis=2), backend=h.backend
        )
        out = self.lstm.forward(out, cache=cache)
        out = Tensor(
            xp.concatenate((self.peeky_h, out.data), axis=2), backend=h.backend
        )
        return self.affine.forward(out, cache=cache)

    def backward_manual(self, dscore: Tensor) -> Tensor:
        if self.peeky_h is None:
            raise RuntimeError("forward must be called before backward")
        h = self.hidden_size
        dout = self.affine.backward(dscore)
        d_lstm_out, d_peeky_out = dout.data[:, :, h:], dout.data[:, :, :h]
        dout = self.lstm.backward(Tensor(d_lstm_out, backend=dscore.backend))
        d_embed, d_peeky_in = dout.data[:, :, h:], dout.data[:, :, :h]
        self.embed.backward(Tensor(d_embed, backend=dscore.backend))
        return Tensor(
            self.lstm.dh.data + (d_peeky_out + d_peeky_in).sum(axis=1),
            backend=dscore.backend,
        )

    def generate(self, h: Tensor, start_id: int, sample_size: int) -> list[int]:
        return _host_sampled_ids(
            h.backend, self.generate_device(h, start_id, sample_size)
        )

    def generate_device(self, h: Tensor, start_id: int, sample_size: int):
        xp = h.backend.xp
        self.lstm.set_state(h)
        batch_size = h.shape[0]
        sample_id = xp.full((batch_size,), start_id, dtype=xp.int64)
        sampled = []
        peeky_h = h.data[:, None, :]
        for _ in range(sample_size):
            x = Tensor(sample_id.reshape((batch_size, 1)), backend=h.backend)
            out = self.embed.forward(x)
            out = Tensor(xp.concatenate((peeky_h, out.data), axis=2), backend=h.backend)
            out = self.lstm.forward(out)
            score = self.affine.forward(
                Tensor(xp.concatenate((peeky_h, out.data), axis=2), backend=h.backend)
            )
            sample_id = score.data[:, -1, :].argmax(axis=1)
            sampled.append(sample_id)
        return _stack_sampled_ids_device(h.backend, sampled)


class PeekySeq2seq(Seq2seq):
    def __init__(
        self,
        vocab_size: int,
        wordvec_size: int,
        hidden_size: int,
        *,
        backend: Backend | str | None = None,
    ) -> None:
        Layer.__init__(self, backend)
        self.encoder = Encoder(
            vocab_size, wordvec_size, hidden_size, backend=self._backend
        )
        self.decoder = PeekyDecoder(
            vocab_size, wordvec_size, hidden_size, backend=self._backend
        )


class AttentionEncoder(Encoder):
    def forward_manual(self, xs: Tensor, *, cache: bool = True) -> Tensor:
        return self.lstm.forward(
            self.embed.forward(xs, cache=cache),
            cache=cache,
        )

    def backward_manual(self, dhs: Tensor) -> None:
        self.embed.backward(self.lstm.backward(dhs))


class AttentionDecoder(Layer):
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
            wordvec_size, hidden_size, stateful=True, backend=self._backend
        )
        self.attention = TimeAttention(backend=self._backend)
        self.affine = TimeAffine(
            hidden_size * 2,
            vocab_size,
            backend=self._backend,
            weight_scale=1 / (hidden_size * 2) ** 0.5,
        )

    def forward_manual(
        self,
        xs: Tensor,
        enc_hs: Tensor,
        *,
        cache: bool = True,
    ) -> Tensor:
        self.lstm.set_state(enc_hs[:, -1, :])
        dec_hs = self.lstm.forward(
            self.embed.forward(xs, cache=cache),
            cache=cache,
        )
        context = self.attention.forward(enc_hs, dec_hs, cache=cache)
        xp = xs.backend.xp
        return self.affine.forward(
            Tensor(
                xp.concatenate((context.data, dec_hs.data), axis=2),
                backend=xs.backend,
            ),
            cache=cache,
        )

    def backward_manual(self, dscore: Tensor) -> Tensor:
        dout = self.affine.backward(dscore)
        hidden = dout.shape[2] // 2
        denc, ddec = self.attention.backward(
            Tensor(dout.data[:, :, :hidden], backend=dscore.backend)
        )
        dx = self.lstm.backward(
            Tensor(dout.data[:, :, hidden:] + ddec.data, backend=dscore.backend)
        )
        denc.data[:, -1, :] += self.lstm.dh.data
        self.embed.backward(dx)
        return denc

    def generate(self, enc_hs: Tensor, start_id: int, sample_size: int) -> list[int]:
        return _host_sampled_ids(
            enc_hs.backend,
            self.generate_device(enc_hs, start_id, sample_size),
        )

    def generate_device(self, enc_hs: Tensor, start_id: int, sample_size: int):
        xp = enc_hs.backend.xp
        self.lstm.set_state(enc_hs[:, -1, :])
        batch_size = enc_hs.shape[0]
        sample_id = xp.full((batch_size,), start_id, dtype=xp.int64)
        sampled = []
        for _ in range(sample_size):
            out = self.embed.forward(
                Tensor(sample_id.reshape((batch_size, 1)), backend=enc_hs.backend)
            )
            dec_hs = self.lstm.forward(out)
            context = self.attention.forward(enc_hs, dec_hs)
            score = self.affine.forward(
                Tensor(
                    xp.concatenate((context.data, dec_hs.data), axis=2),
                    backend=enc_hs.backend,
                )
            )
            sample_id = score.data[:, -1, :].argmax(axis=1)
            sampled.append(sample_id)
        return _stack_sampled_ids_device(enc_hs.backend, sampled)


class AttentionSeq2seq(Seq2seq):
    def __init__(
        self,
        vocab_size: int,
        wordvec_size: int,
        hidden_size: int,
        *,
        backend: Backend | str | None = None,
    ) -> None:
        Layer.__init__(self, backend)
        self.encoder = AttentionEncoder(
            vocab_size, wordvec_size, hidden_size, backend=self._backend
        )
        self.decoder = AttentionDecoder(
            vocab_size, wordvec_size, hidden_size, backend=self._backend
        )

    def generate(self, xs: Tensor, start_id: int, sample_size: int) -> list[int]:
        return _host_sampled_ids(
            xs.backend, self.generate_device(xs, start_id, sample_size)
        )

    def generate_device(self, xs: Tensor, start_id: int, sample_size: int):
        hs = self.encoder.forward(xs)
        return self.decoder.generate_device(hs, start_id, sample_size)


class AttentionPeekyDecoder(Layer):
    """Attention decoder that also peeks at the encoder's final state.

    The final encoder state is concatenated to every decoder input and output,
    while attention supplies the encoder-wide context at every decoder step.
    """

    def __init__(
        self,
        vocab_size: int,
        wordvec_size: int,
        hidden_size: int,
        *,
        backend: Backend | str | None = None,
    ) -> None:
        super().__init__(backend)
        self.hidden_size = hidden_size
        self.embed = TimeEmbedding(vocab_size, wordvec_size, backend=self._backend)
        self.lstm = TimeLSTM(
            wordvec_size + hidden_size,
            hidden_size,
            stateful=True,
            backend=self._backend,
        )
        self.attention = TimeAttention(backend=self._backend)
        self.affine = TimeAffine(
            hidden_size * 3,
            vocab_size,
            backend=self._backend,
            weight_scale=1 / (hidden_size * 3) ** 0.5,
        )
        self.peeky_h = None

    def forward_manual(
        self, xs: Tensor, enc_hs: Tensor, *, cache: bool = True
    ) -> Tensor:
        xp = enc_hs.backend.xp
        self.peeky_h = xp.repeat(enc_hs.data[:, -1:, :], xs.shape[1], axis=1)
        self.lstm.set_state(enc_hs[:, -1, :])
        out = self.embed.forward(xs, cache=cache)
        out = self.lstm.forward(
            Tensor(
                xp.concatenate((self.peeky_h, out.data), axis=2), backend=xs.backend
            ),
            cache=cache,
        )
        context = self.attention.forward(enc_hs, out, cache=cache)
        return self.affine.forward(
            Tensor(
                xp.concatenate((self.peeky_h, context.data, out.data), axis=2),
                backend=xs.backend,
            ),
            cache=cache,
        )

    def backward_manual(self, dscore: Tensor) -> Tensor:
        if self.peeky_h is None:
            raise RuntimeError("forward must be called before backward")
        hidden = self.hidden_size
        dout = self.affine.backward(dscore)
        d_peeky_out = dout.data[:, :, :hidden]
        d_context = Tensor(dout.data[:, :, hidden : 2 * hidden], backend=dscore.backend)
        d_dec = Tensor(dout.data[:, :, 2 * hidden :], backend=dscore.backend)
        denc, d_dec_attention = self.attention.backward(d_context)
        d_lstm = self.lstm.backward(
            Tensor(d_dec.data + d_dec_attention.data, backend=dscore.backend),
        )
        d_peeky_in = d_lstm.data[:, :, :hidden]
        self.embed.backward(Tensor(d_lstm.data[:, :, hidden:], backend=dscore.backend))
        denc.data[:, -1, :] += self.lstm.dh.data
        denc.data[:, -1, :] += (d_peeky_out + d_peeky_in).sum(axis=1)
        return denc

    def generate(self, enc_hs: Tensor, start_id: int, sample_size: int) -> list[int]:
        return _host_sampled_ids(
            enc_hs.backend, self.generate_device(enc_hs, start_id, sample_size)
        )

    def generate_device(self, enc_hs: Tensor, start_id: int, sample_size: int):
        xp = enc_hs.backend.xp
        self.lstm.set_state(enc_hs[:, -1, :])
        batch_size = enc_hs.shape[0]
        sample_id = xp.full((batch_size,), start_id, dtype=xp.int64)
        peeky_h = enc_hs.data[:, -1:, :]
        sampled = []
        for _ in range(sample_size):
            out = self.embed.forward(
                Tensor(sample_id.reshape((batch_size, 1)), backend=enc_hs.backend)
            )
            out = self.lstm.forward(
                Tensor(
                    xp.concatenate((peeky_h, out.data), axis=2), backend=enc_hs.backend
                ),
            )
            context = self.attention.forward(enc_hs, out)
            score = self.affine.forward(
                Tensor(
                    xp.concatenate((peeky_h, context.data, out.data), axis=2),
                    backend=enc_hs.backend,
                ),
            )
            sample_id = score.data[:, -1, :].argmax(axis=1)
            sampled.append(sample_id)
        return _stack_sampled_ids_device(enc_hs.backend, sampled)


class AttentionPeekySeq2seq(Seq2seq):
    """Seq2seq combining encoder-state peeking with additive attention."""

    def __init__(
        self,
        vocab_size: int,
        wordvec_size: int,
        hidden_size: int,
        *,
        backend: Backend | str | None = None,
    ) -> None:
        Layer.__init__(self, backend)
        self.encoder = AttentionEncoder(
            vocab_size, wordvec_size, hidden_size, backend=self._backend
        )
        self.decoder = AttentionPeekyDecoder(
            vocab_size, wordvec_size, hidden_size, backend=self._backend
        )

    def generate(self, xs: Tensor, start_id: int, sample_size: int) -> list[int]:
        return _host_sampled_ids(
            xs.backend, self.generate_device(xs, start_id, sample_size)
        )

    def generate_device(self, xs: Tensor, start_id: int, sample_size: int):
        hs = self.encoder.forward(xs)
        return self.decoder.generate_device(hs, start_id, sample_size)


def _stack_sampled_ids_device(backend: Backend, sampled):
    if not sampled:
        return backend.xp.empty((0, 0), dtype=backend.xp.int64)
    return backend.xp.stack(sampled, axis=1)


def _host_sampled_ids(backend: Backend, sampled) -> list[int]:
    if sampled.size == 0:
        return []
    values = backend.to_numpy(sampled)
    if values.ndim == 2:
        if values.shape[0] != 1:
            raise ValueError("generate() expects a single input sequence")
        values = values[0]
    return [int(value) for value in values]


__all__ = [
    "AttentionPeekySeq2seq",
    "AttentionSeq2seq",
    "BetterRnnlm",
    "Decoder",
    "Encoder",
    "PeekySeq2seq",
    "Rnnlm",
    "Seq2seq",
    "VanillaRnnlm",
]
