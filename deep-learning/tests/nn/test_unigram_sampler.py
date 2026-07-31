from __future__ import annotations

import numpy as np

from mlprosection import Tensor
from mlprosection.nn import UnigramSampler
from mlprosection.nn.model.architecture import (
    CBOW,
    CBOWBatchAdapter,
    SkipGram,
    SkipGramBatchAdapter,
)
from mlprosection.nn.objective import NegativeSampling, SoftmaxWithLoss


def _word2vec_forward(
    model,
    objective,
    inputs,
    target,
    *,
    replay_context=None,
    example_count=None,
):
    batch = objective.prepare(target, replay_context=replay_context)
    prediction = model.forward(inputs, candidates=batch.candidates)
    result = objective.forward(
        prediction,
        batch.target,
        replay_context=batch.replay_context,
        example_count=example_count,
    )
    return result


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
        4, negative_samples=16, sampler=sampler, backend="cpu"
    )
    targets = Tensor(np.array([0, 1], dtype=np.int64), backend="cpu")
    result = _word2vec_forward(
        model,
        objective,
        Tensor(np.array([[1, 2], [2, 3]], dtype=np.int64), backend="cpu"),
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
        8, negative_samples=2, sampler=sampler, backend="cpu"
    )
    centers = Tensor(np.array([1, 2], dtype=np.int64), backend="cpu")
    contexts = Tensor(
        np.array([[0, 2, 3], [1, 3, 4]], dtype=np.int64),
        backend="cpu",
    )

    model_x, objective_t = SkipGramBatchAdapter().prepare(contexts, centers)
    result = _word2vec_forward(
        model,
        objective,
        model_x,
        objective_t,
    )
    _word2vec_forward(
        model,
        objective,
        model_x,
        objective_t,
        replay_context=result.replay_context,
    )

    assert calls == [(6, 2)]
    assert model_x.shape == (2,)
    assert objective_t.shape == (2, 3)
    assert result.replay_context.shape == (2, 3, 2)


def test_grouped_skipgram_matches_pair_expansion_loss_and_gradients() -> None:
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
    expanded_model = SkipGram(8, 3, backend="cpu")
    grouped_model = SkipGram(8, 3, backend="cpu")
    grouped_model.W_in.data[...] = expanded_model.W_in.data
    grouped_model.W_out.data[...] = expanded_model.W_out.data
    expanded_objective = NegativeSampling(
        8, negative_samples=2, backend="cpu"
    )
    grouped_objective = NegativeSampling(
        8, negative_samples=2, backend="cpu"
    )
    expanded_x = Tensor(
        np.repeat(centers.data, contexts.shape[1]),
        backend="cpu",
    )
    expanded_t = contexts.reshape(-1)
    grouped_x, grouped_t = SkipGramBatchAdapter().prepare(contexts, centers)

    expanded = _word2vec_forward(
        expanded_model,
        expanded_objective,
        expanded_x,
        expanded_t,
        replay_context=candidates,
    )
    expanded_model.backward(expanded_objective.backward())
    grouped = _word2vec_forward(
        grouped_model,
        grouped_objective,
        grouped_x,
        grouped_t,
        replay_context=candidates,
    )
    grouped_model.backward(grouped_objective.backward())

    assert grouped_x.shape == (2,)
    assert grouped_t.shape == (2, 3)
    assert np.allclose(grouped.loss.data, expanded.loss.data)
    assert np.allclose(grouped_model.W_in.grad, expanded_model.W_in.grad)
    assert np.allclose(grouped_model.W_out.grad, expanded_model.W_out.grad)


def test_negative_sampling_book_loss_sums_candidates_for_cbow() -> None:
    contexts = Tensor(
        np.array([[0, 2, 3], [1, 3, 4]], dtype=np.int64),
        backend="cpu",
    )
    targets = Tensor(np.array([1, 2], dtype=np.int64), backend="cpu")
    candidates = np.array([[4, 5], [5, 6]], dtype=np.int64)
    model = CBOW(8, 3, backend="cpu")
    objective = NegativeSampling(8, negative_samples=2, backend="cpu")
    model_x, objective_t = CBOWBatchAdapter().prepare(contexts, targets)

    standard = _word2vec_forward(
        model,
        objective,
        model_x,
        objective_t,
        replay_context=candidates,
    )
    model.backward(objective.backward())
    standard_input_gradient = model.W_in.grad.copy()
    standard_output_gradient = model.W_out.grad.copy()
    book = _word2vec_forward(
        model,
        objective,
        model_x,
        objective_t,
        replay_context=candidates,
        example_count=len(contexts),
    )
    model.backward(objective.backward())
    book_input_gradient = model.W_in.grad.copy()

    assert book.reporting_loss is not None
    assert np.allclose(book.loss.data, standard.loss.data * 3)
    assert np.allclose(book.reporting_loss.data, standard.loss.data)
    assert np.allclose(book_input_gradient, standard_input_gradient * 3)
    assert np.allclose(model.W_out.grad, standard_output_gradient * 3)


def test_negative_sampling_book_loss_sums_contexts_and_candidates_for_skipgram() -> None:
    contexts = Tensor(
        np.array([[0, 2, 3], [1, 3, 4]], dtype=np.int64),
        backend="cpu",
    )
    targets = Tensor(np.array([1, 2], dtype=np.int64), backend="cpu")
    candidates = np.array(
        [[4, 5], [5, 6], [6, 7], [5, 6], [6, 7], [0, 7]],
        dtype=np.int64,
    )
    model = SkipGram(8, 3, backend="cpu")
    objective = NegativeSampling(8, negative_samples=2, backend="cpu")
    model_x, objective_t = SkipGramBatchAdapter().prepare(contexts, targets)

    standard = _word2vec_forward(
        model,
        objective,
        model_x,
        objective_t,
        replay_context=candidates,
    )
    model.backward(objective.backward())
    standard_input_gradient = model.W_in.grad.copy()
    standard_output_gradient = model.W_out.grad.copy()
    book = _word2vec_forward(
        model,
        objective,
        model_x,
        objective_t,
        replay_context=candidates,
        example_count=len(contexts),
    )
    model.backward(objective.backward())
    book_input_gradient = model.W_in.grad.copy()

    assert book.reporting_loss is not None
    assert np.allclose(book.loss.data, standard.loss.data * 9)
    assert np.allclose(book.reporting_loss.data, standard.loss.data)
    assert np.allclose(book_input_gradient, standard_input_gradient * 9)
    assert np.allclose(model.W_out.grad, standard_output_gradient * 9)


def test_softmax_with_loss_book_objective_sums_skipgram_contexts_only() -> None:
    contexts = Tensor(
        np.array([[0, 2, 3], [1, 3, 4]], dtype=np.int64),
        backend="cpu",
    )
    targets = Tensor(np.array([1, 2], dtype=np.int64), backend="cpu")
    model = SkipGram(8, 3, backend="cpu")
    objective = SoftmaxWithLoss(backend="cpu")
    model_x = Tensor(
        np.repeat(targets.data, contexts.shape[1]),
        backend="cpu",
    )
    objective_t = contexts.reshape(-1)

    standard = _word2vec_forward(
        model,
        objective,
        model_x,
        objective_t,
    )
    model.backward(objective.backward())
    standard_input_gradient = model.W_in.grad.copy()
    standard_output_gradient = model.W_out.grad.copy()
    book = _word2vec_forward(
        model,
        objective,
        model_x,
        objective_t,
        example_count=len(contexts),
    )
    model.backward(objective.backward())
    book_input_gradient = model.W_in.grad.copy()

    assert book.reporting_loss is not None
    assert np.allclose(book.loss.data, standard.loss.data * 3)
    assert np.allclose(book.reporting_loss.data, standard.loss.data)
    assert np.allclose(book_input_gradient, standard_input_gradient * 3)
    assert np.allclose(model.W_out.grad, standard_output_gradient * 3)


def test_grouped_skipgram_softmax_matches_pair_expansion() -> None:
    centers = Tensor(np.array([1, 2], dtype=np.int64), backend="cpu")
    contexts = Tensor(
        np.array([[0, 2, 2], [1, 3, 4]], dtype=np.int64),
        backend="cpu",
    )
    expanded_model = SkipGram(8, 3, backend="cpu")
    grouped_model = SkipGram(8, 3, backend="cpu")
    grouped_model.W_in.data[...] = expanded_model.W_in.data
    grouped_model.W_out.data[...] = expanded_model.W_out.data

    expanded_x = Tensor(
        np.repeat(centers.data, contexts.shape[1]),
        backend="cpu",
    )
    expanded_t = contexts.reshape(-1)
    grouped_x, grouped_t = SkipGramBatchAdapter().prepare(
        contexts,
        centers,
    )
    expanded_objective = SoftmaxWithLoss(backend="cpu")
    grouped_objective = SoftmaxWithLoss(
        grouped_targets=True,
        backend="cpu",
    )

    expanded = _word2vec_forward(
        expanded_model,
        expanded_objective,
        expanded_x,
        expanded_t,
        example_count=len(contexts),
    )
    expanded_model.backward(expanded_objective.backward())
    grouped = _word2vec_forward(
        grouped_model,
        grouped_objective,
        grouped_x,
        grouped_t,
        example_count=len(contexts),
    )
    grouped_model.backward(grouped_objective.backward())

    assert expanded_x.shape == (6,)
    assert expanded_t.shape == (6,)
    assert grouped_x.shape == (2,)
    assert grouped_t.shape == (2, 3)
    assert grouped.unit_count == expanded.unit_count == 6
    assert grouped.reporting_loss is not None
    assert expanded.reporting_loss is not None
    assert np.allclose(grouped.loss.data, expanded.loss.data)
    assert np.allclose(
        grouped.reporting_loss.data,
        expanded.reporting_loss.data,
    )
    assert np.allclose(grouped_model.W_in.grad, expanded_model.W_in.grad)
    assert np.allclose(grouped_model.W_out.grad, expanded_model.W_out.grad)
