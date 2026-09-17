"""Peeky sequence-to-sequence decoder and model architectures."""

from __future__ import annotations

from deepscratch.core import Tensor
from deepscratch.core.backend import Backend
from deepscratch.nn.layers import Layer, TimeAffine, TimeEmbedding, TimeLSTM

from .sampling import _host_sampled_ids, _stack_sampled_ids_device
from .seq2seq import Decoder, Encoder, Seq2seq


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


__all__ = [
    "PeekyDecoder",
    "PeekySeq2seq",
]
