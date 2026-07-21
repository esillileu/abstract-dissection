from __future__ import annotations

import numpy as np

from mlprosection import Tensor
from mlprosection.nn import UnigramSampler
from mlprosection.nn.model import Word2Vec
from mlprosection.optim.SGD import Adam
from mlprosection.trainer import BookWord2VecTrainer


def test_negative_sampling_sum_reduction_matches_book_prediction_term_scale() -> None:
    sampler = UnigramSampler.uniform(5, backend="cpu")
    sampler.sample = lambda targets, *, sample_size: np.full((len(targets), sample_size), 4, dtype=np.int64)  # type: ignore[method-assign]
    inputs = Tensor(np.array([[0, 1], [1, 2]], dtype=np.int64), backend="cpu")
    targets = Tensor(np.array([2, 3], dtype=np.int64), backend="cpu")
    mean_model = Word2Vec(5, 3, negative_samples=2, sampler=sampler, backend="cpu")
    sum_model = Word2Vec(5, 3, negative_samples=2, loss_reduction="sum", sampler=sampler, backend="cpu")
    sum_model.W_in.data[...] = mean_model.W_in.data
    sum_model.W_out.data[...] = mean_model.W_out.data

    mean_loss = mean_model.forward(inputs, targets)
    sum_loss = sum_model.forward(inputs, targets)
    mean_model.backward()
    sum_model.backward()

    assert np.isclose(float(sum_loss.data), float(mean_loss.data) * 3)
    assert np.allclose(sum_model.W_in.grad, mean_model.W_in.grad * 3)
    assert np.allclose(sum_model.W_out.grad, mean_model.W_out.grad * 3)


def test_book_word2vec_trainer_discards_partial_batch_and_logs_first_update() -> None:
    inputs = Tensor(np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]], dtype=np.int64), backend="cpu")
    targets = Tensor(np.array([2, 3, 4, 5, 0], dtype=np.int64), backend="cpu")
    model = Word2Vec(6, 3, objective="full_softmax", backend="cpu")
    trainer = BookWord2VecTrainer(
        model,
        Adam(list(model.named_parameters())),
        max_epoch=1,
        batch_size=2,
        log_interval=2,
        prediction_term_count=1,
    )

    history = trainer.fit(inputs, targets)

    assert trainer.global_step == 2
    assert len(history.interval_loss) == 1
    assert trainer.logs.train[0]["global_step"] == 1
    assert trainer.logs.train[0]["iteration"] == 1
