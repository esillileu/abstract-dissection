from __future__ import annotations

import numpy as np

from mlprosection import Tensor
from mlprosection.nn.layers import TimeSoftmaxWithLoss
from mlprosection.nn.model.recurrent import BetterRnnlm, Rnnlm, Seq2seq


def test_rnnlm_supports_external_criterion_interface() -> None:
    model = Rnnlm(vocab_size=7, wordvec_size=4, hidden_size=5, backend="cpu")
    criterion = TimeSoftmaxWithLoss()
    xs = Tensor(np.array([[0, 1, 2], [3, 4, 5]]), backend="cpu")
    ts = Tensor(np.array([[1, 2, 3], [4, 5, 6]]), backend="cpu")

    scores = model.forward(xs)
    loss = criterion.forward(scores, ts)
    dout = criterion.backward()
    model.backward(dout)

    assert scores.shape == (2, 3, 7)
    assert loss.shape == ()
    assert all(param.grad is not None for _, param in model.named_parameters())


def test_rnnlm_supports_b2_internal_loss_call() -> None:
    model = Rnnlm(vocab_size=7, wordvec_size=4, hidden_size=5, backend="cpu")
    xs = Tensor(np.array([[0, 1, 2], [3, 4, 5]]), backend="cpu")
    ts = Tensor(np.array([[1, 2, 3], [4, 5, 6]]), backend="cpu")

    loss = model.forward(xs, ts)
    model.backward()

    assert loss.shape == ()


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

    loss = model.forward(xs, ts)
    model.backward()
    params = list(model.named_parameters())

    assert loss.shape == ()
    assert model.layers[-1].W is model.embed.W
    assert len([param for _, param in params if param is model.embed.W]) == 1


def test_seq2seq_forward_backward_generate() -> None:
    model = Seq2seq(vocab_size=10, wordvec_size=4, hidden_size=5, backend="cpu")
    xs = Tensor(np.array([[1, 2, 3], [4, 5, 6]]), backend="cpu")
    ts = Tensor(np.array([[0, 1, 2, 3], [0, 4, 5, 6]]), backend="cpu")

    loss = model.forward(xs, ts)
    model.backward()
    sampled = model.generate(xs[:1], start_id=0, sample_size=4)

    assert loss.shape == ()
    assert len(sampled) == 4
    assert all(isinstance(sample_id, int) for sample_id in sampled)
