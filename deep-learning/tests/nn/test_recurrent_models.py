from __future__ import annotations

import numpy as np

from mlprosection import Tensor
from mlprosection.nn.model.architecture import BetterRnnlm, Rnnlm, Seq2seq, TiedRnnlm
from mlprosection.nn.objective import TemporalSoftmaxCrossEntropy


def test_rnnlm_supports_external_objective_interface() -> None:
    model = Rnnlm(vocab_size=7, wordvec_size=4, hidden_size=5, backend="cpu")
    objective = TemporalSoftmaxCrossEntropy()
    xs = Tensor(np.array([[0, 1, 2], [3, 4, 5]]), backend="cpu")
    ts = Tensor(np.array([[1, 2, 3], [4, 5, 6]]), backend="cpu")

    scores = model.forward(xs)
    result = objective.forward(scores, ts)
    dout = objective.backward()
    model.backward(dout)

    assert scores.shape == (2, 3, 7)
    assert result.loss.shape == ()
    assert all(param.grad is not None for _, param in model.named_parameters())


def test_better_rnnlm_ties_embedding_and_affine_weight() -> None:
    model = BetterRnnlm(
        vocab_size=7,
        wordvec_size=5,
        hidden_size=5,
        dropout_ratio=0.0,
        backend="cpu",
    )
    xs = Tensor(np.array([[0, 1, 2], [3, 4, 5]]), backend="cpu")
    ts = Tensor(np.array([[1, 2, 3], [4, 5, 6]]), backend="cpu")

    objective = TemporalSoftmaxCrossEntropy()
    result = objective.forward(model.forward(xs), ts)
    model.backward(objective.backward())
    params = list(model.named_parameters())

    assert result.loss.shape == ()
    assert model.layers[-1].W is model.embed.W
    assert len([param for _, param in params if param is model.embed.W]) == 1


def test_tied_rnnlm_ties_single_lstm_embedding_and_affine_weight() -> None:
    model = TiedRnnlm(
        vocab_size=7,
        wordvec_size=5,
        hidden_size=5,
        backend="cpu",
    )
    xs = Tensor(np.array([[0, 1, 2], [3, 4, 5]]), backend="cpu")
    ts = Tensor(np.array([[1, 2, 3], [4, 5, 6]]), backend="cpu")

    objective = TemporalSoftmaxCrossEntropy()
    objective.forward(model.forward(xs), ts)
    model.backward(objective.backward())

    assert model.layers[-1].W is model.embed.W
    assert len([param for _, param in model.named_parameters() if param is model.embed.W]) == 1
    assert np.any(model.embed.W.grad != 0)


def test_seq2seq_forward_backward_generate() -> None:
    model = Seq2seq(vocab_size=10, wordvec_size=4, hidden_size=5, backend="cpu")
    xs = Tensor(np.array([[1, 2, 3], [4, 5, 6]]), backend="cpu")
    ts = Tensor(np.array([[0, 1, 2, 3], [0, 4, 5, 6]]), backend="cpu")

    objective = TemporalSoftmaxCrossEntropy()
    scores = model.forward(xs, ts[:, :-1])
    result = objective.forward(scores, ts[:, 1:])
    model.backward(objective.backward())
    sampled = model.generate(xs[:1], start_id=0, sample_size=4)

    assert result.loss.shape == ()
    assert len(sampled) == 4
    assert all(isinstance(sample_id, int) for sample_id in sampled)


def test_seq2seq_generate_device_supports_batched_inputs() -> None:
    model = Seq2seq(vocab_size=10, wordvec_size=4, hidden_size=5, backend="cpu")
    xs = Tensor(np.array([[1, 2, 3], [4, 5, 6]]), backend="cpu")

    sampled = model.generate_device(xs, start_id=0, sample_size=4)

    assert sampled.shape == (2, 4)


def test_rnnlm_cache_free_forward_preserves_loss_without_backward_caches() -> None:
    model = Rnnlm(vocab_size=7, wordvec_size=4, hidden_size=5, backend="cpu")
    xs = Tensor(np.array([[0, 1, 2], [3, 4, 5]]), backend="cpu")
    model.reset_runtime_state()
    expected = model.forward(xs).data.copy()
    model.reset_runtime_state()
    actual = model.forward(xs, cache=False)

    np.testing.assert_array_equal(actual.data, expected)
    assert model.lstm_layer.layers == []
    assert model.layers[0].layer.idx is None
    assert model.layers[-1].layer.x is None


def test_seq2seq_cache_free_forward_preserves_loss_without_recurrent_caches() -> None:
    model = Seq2seq(vocab_size=10, wordvec_size=4, hidden_size=5, backend="cpu")
    xs = Tensor(np.array([[1, 2, 3], [4, 5, 6]]), backend="cpu")
    ts = Tensor(np.array([[0, 1, 2, 3], [0, 4, 5, 6]]), backend="cpu")

    expected = model.forward(xs, ts[:, :-1]).data.copy()
    actual = model.forward(xs, ts[:, :-1], cache=False)

    np.testing.assert_array_equal(actual.data, expected)
    assert model.encoder.lstm.layers == []
    assert model.decoder.lstm.layers == []
