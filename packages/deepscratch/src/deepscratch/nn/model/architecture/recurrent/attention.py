"""Attention sequence-to-sequence encoder, decoder, and model architectures."""

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

from .sampling import _host_sampled_ids, _stack_sampled_ids_device
from .seq2seq import Encoder, Seq2seq


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


__all__ = [
    "AttentionDecoder",
    "AttentionEncoder",
    "AttentionSeq2seq",
]
