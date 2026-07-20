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
