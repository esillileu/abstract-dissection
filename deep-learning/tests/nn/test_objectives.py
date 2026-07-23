from __future__ import annotations

import numpy as np
import pytest

from mlprosection import Tensor
from mlprosection.nn.functional import (
    binary_cross_entropy_with_logits,
    softmax_cross_entropy,
)
from mlprosection.nn.objective import (
    BinaryCrossEntropyWithLogits,
    SoftmaxCrossEntropy,
    TemporalSoftmaxCrossEntropy,
)


def test_softmax_cross_entropy_function_returns_loss_gradient_and_count() -> None:
    logits = Tensor(
        np.array([[1.0, 2.0, 0.0], [0.5, -0.5, 1.0]]),
        backend="cpu",
    )
    target = Tensor(np.array([1, 2]), backend="cpu")

    computation = softmax_cross_entropy(logits, target)

    assert computation.loss.shape == ()
    assert computation.gradient.shape == logits.shape
    assert computation.unit_count == 2
    np.testing.assert_allclose(
        computation.gradient.data.sum(axis=-1),
        np.zeros(2),
        atol=1e-7,
    )


def test_temporal_softmax_ignores_labels_in_loss_gradient_and_count() -> None:
    logits = Tensor(
        np.array(
            [
                [[2.0, 1.0, 0.0], [0.0, 2.0, 1.0]],
                [[1.0, 0.0, 2.0], [2.0, 0.0, 1.0]],
            ]
        ),
        backend="cpu",
    )
    target = Tensor(np.array([[0, -1], [2, 0]]), backend="cpu")
    objective = TemporalSoftmaxCrossEntropy(ignore_label=-1)

    result = objective.forward(logits, target)
    gradient = objective.backward()

    assert result.unit_count == 3
    np.testing.assert_array_equal(gradient.data[0, 1], np.zeros(3))


def test_objective_cache_false_does_not_replace_backward_gradient() -> None:
    objective = SoftmaxCrossEntropy()
    training_logits = Tensor(np.array([[1.0, 2.0, 0.0]]), backend="cpu")
    probe_logits = Tensor(np.array([[2.0, 0.0, 1.0]]), backend="cpu")
    target = Tensor(np.array([1]), backend="cpu")

    objective.forward(training_logits, target)
    expected = objective.backward().data.copy()
    objective.forward(probe_logits, target, cache=False)

    np.testing.assert_allclose(objective.backward().data, expected)


def test_binary_cross_entropy_function_and_objective_share_computation() -> None:
    logits = Tensor(np.array([[0.0, 1.0], [-1.0, 2.0]]), backend="cpu")
    target = Tensor(np.array([[0.0, 1.0], [1.0, 0.0]]), backend="cpu")
    computation = binary_cross_entropy_with_logits(logits, target)
    objective = BinaryCrossEntropyWithLogits()

    result = objective.forward(logits, target)

    np.testing.assert_allclose(result.loss.data, computation.loss.data)
    np.testing.assert_allclose(objective.backward().data, computation.gradient.data)
    assert result.unit_count == target.size


def test_objective_backward_requires_cached_forward() -> None:
    with pytest.raises(RuntimeError, match=r"forward\(cache=True\)"):
        BinaryCrossEntropyWithLogits().backward()
