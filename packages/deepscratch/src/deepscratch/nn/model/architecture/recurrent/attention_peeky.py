"""Attention peeky sequence-to-sequence decoder and model architectures."""

from __future__ import annotations

from deepscratch.core import Tensor
from deepscratch.core.backend import Backend
from deepscratch.nn.layers import (
    Layer,
    TimeAffine,
    TimeAttention,
    TimeEmbedding,
    TimeLSTM,
)

from .attention import AttentionEncoder
from .sampling import _host_sampled_ids, _stack_sampled_ids_device
from .seq2seq import Seq2seq


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


__all__ = [
    "AttentionPeekyDecoder",
    "AttentionPeekySeq2seq",
]
