from __future__ import annotations

import numpy as np

from mlprosection import Tensor
from mlprosection.nn import UnigramSampler
from mlprosection.nn.model import Word2Vec


def test_unigram_sampler_builds_powered_distribution_once_and_excludes_targets() -> None:
    corpus = np.array([0] * 80 + [1] * 15 + [2] * 5, dtype=np.int64)
    sampler = UnigramSampler.from_corpus(
        corpus, vocab_size=3, backend="cpu", power=1.0, rejection_rounds=4
    )

    drawn = sampler._draw((30_000, 1)).reshape(-1)
    frequencies = np.bincount(drawn, minlength=3) / len(drawn)
    assert np.allclose(frequencies, [0.8, 0.15, 0.05], atol=0.015)

    targets = Tensor(np.array([0, 1, 2], dtype=np.int64), backend="cpu")
    negatives = sampler.sample(targets, sample_size=128)
    assert negatives.shape == (3, 128)
    assert not np.any(negatives == targets.data[:, None])
    assert sampler.metadata == {
        "algorithm": "alias_target_rejection_v1",
        "power": 1.0,
        "replacement": True,
        "excludes_positive": True,
        "rejection_rounds": 4,
    }


def test_word2vec_uses_unigram_sampler_for_negative_candidates() -> None:
    sampler = UnigramSampler.from_corpus(
        np.array([0] * 5 + [1] * 4 + [2] * 3 + [3] * 2),
        vocab_size=4,
        backend="cpu",
    )
    model = Word2Vec(4, 3, negative_samples=16, sampler=sampler, backend="cpu")
    targets = Tensor(np.array([0, 1], dtype=np.int64), backend="cpu")
    model.forward(Tensor(np.array([[1, 2], [2, 3]], dtype=np.int64), backend="cpu"), targets)

    _hidden, candidates, _source, _values = model.cache[-1]
    assert np.array_equal(candidates[:, 0], targets.data)
    assert not np.any(candidates[:, 1:] == targets.data[:, None])


def test_skipgram_samples_all_context_targets_in_one_call() -> None:
    sampler = UnigramSampler.uniform(8, backend="cpu")
    calls = []
    sample = sampler.sample

    def counted_sample(targets, *, sample_size):
        calls.append((len(targets), sample_size))
        return sample(targets, sample_size=sample_size)

    sampler.sample = counted_sample
    model = Word2Vec(
        8, 3, architecture="skipgram", objective="negative_sampling",
        negative_samples=2, sampler=sampler, backend="cpu",
    )
    centers = Tensor(np.array([1, 2], dtype=np.int64), backend="cpu")
    contexts = Tensor(
        np.array([[0, 2, 3], [1, 3, 4]], dtype=np.int64),
        backend="cpu",
    )

    model.forward(centers, contexts)
    candidates = model.last_negative_candidates()
    model.forward(centers, contexts, negative_candidates=candidates)

    assert calls == [(6, 2)]
    assert len(model.cache) == 1
    assert np.array_equal(model.cache[0][1][:, 0], contexts.data.reshape(-1))


def test_vectorized_skipgram_preserves_per_context_loss_and_gradients() -> None:
    centers = Tensor(np.array([1, 2], dtype=np.int64), backend="cpu")
    contexts = Tensor(
        np.array([[0, 2, 3], [1, 3, 4]], dtype=np.int64),
        backend="cpu",
    )
    candidates = [
        np.array([[4, 5], [5, 6]], dtype=np.int64),
        np.array([[5, 6], [6, 7]], dtype=np.int64),
        np.array([[6, 7], [0, 7]], dtype=np.int64),
    ]
    vectorized = Word2Vec(
        8, 3, architecture="skipgram", objective="negative_sampling",
        negative_samples=2, backend="cpu",
    )
    reference = Word2Vec(
        8, 3, architecture="skipgram", objective="negative_sampling",
        negative_samples=2, backend="cpu",
    )
    reference.W_in.data[...] = vectorized.W_in.data
    reference.W_out.data[...] = vectorized.W_out.data

    loss = vectorized.forward(centers, contexts, negative_candidates=candidates)
    vectorized.backward()

    reference.cache = []
    losses = [
        reference._objective(
            reference.W_in.data[centers.data], contexts.data[:, column],
            centers.data, fixed_candidates=candidates[column],
        )
        for column in range(contexts.shape[1])
    ]
    reference_loss = sum(losses) / len(losses)
    reference.backward()

    assert np.allclose(loss.data, reference_loss)
    assert np.allclose(vectorized.W_in.grad, reference.W_in.grad)
    assert np.allclose(vectorized.W_out.grad, reference.W_out.grad)
