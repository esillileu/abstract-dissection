"""Unit tests for DS1 DeepScratch representation adapters."""

from __future__ import annotations

import numpy as np
import pytest
from deepscratch.nn.model.architecture import MLP, DeepCNN, SimpleCNN, TwoLayerNet
from deepscratch.nn.objective import SoftmaxCrossEntropy
from deepscratch.optim.SGD import SGD, AdaGrad, Adam, Momentum
from deepscratch.optim.transform import L2Regularization

from dlfs.ds1.implemented.adapters import (
    activation_fn,
    book_gradients,
    build_ds1_model,
    build_ds1_objective,
    build_ds1_optimizer,
    initializer_scale,
    training_parameters,
)


def test_build_ds1_model_instantiates_expected_architectures() -> None:
    two_layer = build_ds1_model(
        {"name": "TwoLayerNet", "input_size": 10, "hidden_size": 5, "output_size": 2}
    )
    assert isinstance(two_layer, TwoLayerNet)

    mlp = build_ds1_model(
        {
            "name": "MLP",
            "input_size": 10,
            "hidden_sizes": [8, 6],
            "output_size": 2,
            "activation": "relu",
            "use_batchnorm": False,
        }
    )
    assert isinstance(mlp, MLP)

    simple_cnn = build_ds1_model(
        {
            "name": "SimpleCNN",
            "input_channels": 1,
            "image_size": 28,
            "num_classes": 10,
            "conv_channels": 16,
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
            "hidden_size": 32,
        }
    )
    assert isinstance(simple_cnn, SimpleCNN)

    deep_cnn = build_ds1_model(
        {
            "name": "DeepCNN",
            "input_channels": 1,
            "image_size": 28,
            "num_classes": 10,
            "channels": [16, 32, 64],
            "hidden_size": 50,
        }
    )
    assert isinstance(deep_cnn, DeepCNN)


def test_build_ds1_objective() -> None:
    obj = build_ds1_objective(
        {"name": "SoftmaxCrossEntropy", "reduction": "mean"}, "cpu"
    )
    assert isinstance(obj, SoftmaxCrossEntropy)
    assert obj.reduction == "mean"


def test_build_ds1_optimizer_and_l2_regularization() -> None:
    two_layer = TwoLayerNet(input_size=4, hidden_size=3, output_size=2)
    obj = SoftmaxCrossEntropy()
    params = training_parameters(two_layer, obj)

    sgd = build_ds1_optimizer(
        {"name": "sgd", "learning_rate": 0.05, "weight_decay": 0.01}, params
    )
    assert isinstance(sgd, SGD)
    assert sgd.lr == 0.05
    assert len(sgd.pre_step_hooks) == 1
    assert isinstance(sgd.pre_step_hooks[0], L2Regularization)

    momentum = build_ds1_optimizer(
        {"name": "momentum", "learning_rate": 0.01, "momentum": 0.95}, params
    )
    assert isinstance(momentum, Momentum)
    assert momentum.m == 0.95

    adagrad = build_ds1_optimizer(
        {"name": "adagrad", "learning_rate": 0.01, "eps": 1e-6}, params
    )
    assert isinstance(adagrad, AdaGrad)
    assert adagrad.eps == 1e-6

    adam = build_ds1_optimizer(
        {"name": "adam", "learning_rate": 0.002, "beta1": 0.85}, params
    )
    assert isinstance(adam, Adam)
    assert adam.beta1 == 0.85


def test_initializer_scale_and_activation_fn() -> None:
    assert initializer_scale("he", 100) == pytest.approx(np.sqrt(2.0 / 100))
    assert initializer_scale("xavier", 100) == pytest.approx(np.sqrt(1.0 / 100))
    assert initializer_scale("std:0.01", 100) == 0.01

    x = np.array([-1.0, 0.0, 2.0])
    np.testing.assert_allclose(activation_fn(x, "relu"), [0.0, 0.0, 2.0])
    np.testing.assert_allclose(activation_fn(x, "sigmoid"), 1.0 / (1.0 + np.exp(-x)))
    np.testing.assert_allclose(activation_fn(x, "tanh"), np.tanh(x))


def test_book_gradients_extraction() -> None:
    two_layer = TwoLayerNet(input_size=4, hidden_size=3, output_size=2)
    grads = book_gradients(two_layer)
    assert set(grads.keys()) == {"W1", "b1", "W2", "b2"}
    assert grads["W1"].shape == (4, 3)
    assert grads["b1"].shape == (3,)
    assert grads["W2"].shape == (3, 2)
    assert grads["b2"].shape == (2,)
