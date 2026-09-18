from __future__ import annotations

import numpy as np
from deepscratch.core import Tensor
from deepscratch.nn.model.architecture import AttentionSeq2seq


def _attention_example_ids(*, size: int, count: int, seed: int) -> list[int]:
    if size < 1:
        return []
    rng = np.random.RandomState(seed)
    return [int(rng.randint(0, size)) for _ in range(count)]


def _generate_attention_with_weights(
    model: AttentionSeq2seq, question: Tensor, start_id: int, sample_size: int, backend
) -> tuple[list[int], np.ndarray]:
    xp = backend.xp
    was_training = bool(getattr(model, "training", True))
    model.train(False)
    try:
        enc_hs = model.encoder.forward(question)
        model.decoder.lstm.set_state(enc_hs[:, -1, :])
        sample_id = xp.asarray(start_id, dtype=xp.int64)
        sampled = []
        weights = []
        for _ in range(sample_size):
            out = model.decoder.embed.forward(
                Tensor(sample_id.reshape((1, 1)), backend=backend)
            )
            dec_hs = model.decoder.lstm.forward(out)
            context = model.decoder.attention.forward(enc_hs, dec_hs)
            weights.append(model.decoder.attention.weights[0, 0].copy())
            score = model.decoder.affine.forward(
                Tensor(
                    xp.concatenate((context.data, dec_hs.data), axis=2), backend=backend
                )
            )
            sample_id = score.data.reshape(-1).argmax()
            sampled.append(sample_id)
        host_ids = (
            backend.to_numpy(xp.stack(sampled))
            if sampled
            else np.asarray([], dtype=np.int64)
        )
        host_weights = (
            backend.to_numpy(xp.stack(weights)) if weights else np.empty((0, 0))
        )
        return [int(value) for value in host_ids], np.asarray(host_weights)
    finally:
        model.train(was_training)


def _teacher_forced_attention_with_weights(
    model: AttentionSeq2seq,
    question: Tensor,
    target: np.ndarray,
    backend,
) -> tuple[list[int], np.ndarray]:
    """Run the book's teacher-forced decoder once and retain every score/weight."""
    was_training = bool(getattr(model, "training", True))
    model.train(False)
    try:
        decoder_x = Tensor(
            backend.xp.asarray(target[:-1][None, :], dtype=backend.xp.int64),
            backend=backend,
        )
        scores = model.forward(question, decoder_x, cache=True)
        weights = backend.to_numpy(model.decoder.attention.weights[0])
        predicted = backend.to_numpy(scores.data[0].argmax(axis=1))
        return [int(value) for value in predicted], np.asarray(weights)
    finally:
        model.train(was_training)
