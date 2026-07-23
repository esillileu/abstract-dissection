from __future__ import annotations

import numpy as np

from mlprosection import Tensor
from mlprosection.nn import UnigramSampler
from mlprosection.nn.model.architecture import CBOW, SkipGram, SkipGramBatchAdapter
from mlprosection.nn.objective import NegativeSampling


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
    model = CBOW(4, 3, backend="cpu")
    objective = NegativeSampling(
        4, 3, negative_samples=16, sampler=sampler, backend="cpu"
    )
    targets = Tensor(np.array([0, 1], dtype=np.int64), backend="cpu")
    result = objective.forward(
        model.forward(Tensor(np.array([[1, 2], [2, 3]], dtype=np.int64), backend="cpu")),
        targets,
    )
    assert not np.any(result.replay_context == targets.data[:, None])


def test_conditional_cdf_sampler_excludes_targets_and_reports_algorithm() -> None:
    sampler = UnigramSampler.from_corpus(
        np.array([0] * 80 + [1] * 15 + [2] * 5, dtype=np.int64),
        vocab_size=3,
        backend="cpu",
        power=1.0,
        algorithm=UnigramSampler.CONDITIONAL_CDF,
    )
    targets = Tensor(
        np.repeat(np.arange(3, dtype=np.int64), 10_000),
        backend="cpu",
    )

    drawn = sampler.sample(targets, sample_size=1).reshape(3, 10_000)

    assert not np.any(drawn == np.arange(3)[:, None])
    assert np.allclose(
        np.bincount(drawn[0], minlength=3) / drawn.shape[1],
        [0.0, 0.75, 0.25],
        atol=0.02,
    )
    assert sampler.metadata == {
        "algorithm": "conditional_cdf_target_exclusion_v1",
        "power": 1.0,
        "replacement": True,
        "excludes_positive": True,
        "rejection_rounds": None,
    }


def test_unigram_sampler_rejects_unknown_algorithm() -> None:
    with np.testing.assert_raises_regex(
        ValueError, "unsupported unigram sampling algorithm"
    ):
        UnigramSampler.uniform(3, backend="cpu", algorithm="unknown")


def test_skipgram_samples_all_context_targets_in_one_call() -> None:
    sampler = UnigramSampler.uniform(8, backend="cpu")
    calls = []
    sample = sampler.sample

    def counted_sample(targets, *, sample_size):
        calls.append((len(targets), sample_size))
        return sample(targets, sample_size=sample_size)

    sampler.sample = counted_sample
    model = SkipGram(8, 3, backend="cpu")
    objective = NegativeSampling(
        8, 3, negative_samples=2, sampler=sampler, backend="cpu"
    )
    centers = Tensor(np.array([1, 2], dtype=np.int64), backend="cpu")
    contexts = Tensor(
        np.array([[0, 2, 3], [1, 3, 4]], dtype=np.int64),
        backend="cpu",
    )

    model_x, objective_t = SkipGramBatchAdapter().prepare(contexts, centers)
    result = objective.forward(model.forward(model_x), objective_t)
    objective.forward(
        model.forward(model_x), objective_t,
        replay_context=result.replay_context,
    )

    assert calls == [(6, 2)]
    assert result.replay_context.shape == (6, 2)


def test_vectorized_skipgram_preserves_per_context_loss_and_gradients() -> None:
    centers = Tensor(np.array([1, 2], dtype=np.int64), backend="cpu")
    contexts = Tensor(
        np.array([[0, 2, 3], [1, 3, 4]], dtype=np.int64),
        backend="cpu",
    )
    candidates = np.stack([
        np.array([[4, 5], [5, 6]], dtype=np.int64),
        np.array([[5, 6], [6, 7]], dtype=np.int64),
        np.array([[6, 7], [0, 7]], dtype=np.int64),
    ], axis=1).reshape(6, 2)
    model = SkipGram(8, 3, backend="cpu")
    objective = NegativeSampling(8, 3, negative_samples=2, backend="cpu")
    model_x, objective_t = SkipGramBatchAdapter().prepare(contexts, centers)
    first = objective.forward(
        model.forward(model_x), objective_t, replay_context=candidates
    )
    model.backward(objective.backward())
    input_gradient = model.W_in.grad.copy()
    output_gradient = objective.W_out.grad.copy()
    second = objective.forward(
        model.forward(model_x), objective_t, replay_context=candidates
    )
    model.backward(objective.backward())

    assert np.allclose(first.loss.data, second.loss.data)
    assert np.allclose(input_gradient, model.W_in.grad)
    assert np.allclose(output_gradient, objective.W_out.grad)
