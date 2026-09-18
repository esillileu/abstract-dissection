"""Sequence-to-sequence encoder, decoder, and model architectures."""

from __future__ import annotations

from deepscratch.core import Tensor
from deepscratch.core.backend import Backend
from deepscratch.nn.layers import Layer, TimeAffine, TimeEmbedding, TimeLSTM
from deepscratch.nn.model.base import GenerativeModel

from .sampling import _host_sampled_ids, _stack_sampled_ids_device


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


__all__ = [
    "Decoder",
    "Encoder",
    "Seq2seq",
]
