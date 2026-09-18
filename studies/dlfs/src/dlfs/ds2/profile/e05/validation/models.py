"""Full BetterRnnlm model lockstep and reproducibility validation."""

from __future__ import annotations

import numpy as np
from deepscratch.core import Tensor
from deepscratch.datasets import load_ptb
from deepscratch.nn.model.architecture.recurrent import BetterRnnlm
from deepscratch.optim.SGD import SGD
from deepscratch.optim.transform import ClipGradNorm

from ..phase1 import replace_better_rnnlm_lstms as replace_phase1_lstms
from ..phase2 import replace_better_rnnlm_lstms as replace_phase2_lstms
from ..phase3 import (
    Phase3TemporalSoftmaxCrossEntropy,
    UnfusedTemporalSoftmaxCrossEntropy,
)
from ..reference import replace_better_rnnlm_lstms
from .metrics import FORWARD_CEILING, GRADIENT_CEILING, _within, error_metrics


def _copy_model(source, target) -> None:
    source_params = dict(source.named_parameters())
    target_params = dict(target.named_parameters())
    if source_params.keys() != target_params.keys():
        raise AssertionError("model parameter names differ")
    for name in source_params:
        target_params[name].data[...] = source_params[name].data


def _model(backend, implementation: str, seed: int):
    backend.xp.random.seed(seed)
    model = BetterRnnlm(10_000, 650, 650, 0.5, backend=backend)
    if implementation == "reference":
        replace_better_rnnlm_lstms(model)
    elif implementation == "phase1":
        replace_phase1_lstms(model)
    elif implementation in {"phase2", "phase3"}:
        replace_phase2_lstms(model)
    objective_cls = (
        Phase3TemporalSoftmaxCrossEntropy
        if implementation == "phase3"
        else UnfusedTemporalSoftmaxCrossEntropy
    )
    objective = objective_cls(reduction="mean", backend=backend)
    params = [(name, parameter) for name, parameter in model.named_parameters()]
    optimizer = SGD(params, lr=20.0)
    return model, objective, optimizer, ClipGradNorm(0.25)


def _update(backend, objects, xs, targets, dropout_seed: int):
    model, objective, optimizer, clipper = objects
    backend.xp.random.seed(dropout_seed)
    prediction = model.forward(xs)
    result = objective.forward(prediction, targets)
    model.backward(objective.backward())
    named = [(name, p) for name, p in optimizer.params if p.grad is not None]
    clipper(named)
    for name, parameter in named:
        optimizer.update_one(name, parameter)
    model.detach_runtime_state()
    for layer in model.lstm_layers:
        layer.detach_state()
    backend.synchronize()
    return backend.scalar_to_float(result.loss.data)


def _batch(backend, corpus, update: int):
    xp, batch_size, time_size = backend.xp, 20, 35
    size = len(corpus) - 1
    jump = size // batch_size
    offsets = xp.arange(batch_size) * jump
    positions = (
        offsets[:, None] + update * time_size + xp.arange(time_size)[None, :]
    ) % size
    return (
        Tensor(corpus[positions], backend=backend),
        Tensor(corpus[positions + 1], backend=backend),
    )


def lockstep(backend, *, implementation: str = "phase1") -> dict[str, object]:
    corpus = backend.xp.asarray(load_ptb()["train"], dtype=backend.xp.int64)
    reference = _model(backend, "reference", 314159)
    actual = _model(backend, implementation, 271828)
    _copy_model(reference[0], actual[0])
    rows = []
    passed = True
    for update in range(5):
        xs, targets = _batch(backend, corpus, update)
        expected_loss = _update(backend, reference, xs, targets, 9000 + update)
        actual_loss = _update(backend, actual, xs, targets, 9000 + update)
        param_errors = {}
        for name, parameter in reference[0].named_parameters():
            other = dict(actual[0].named_parameters())[name]
            param_errors[name] = error_metrics(backend, parameter.data, other.data)
        state_errors = {}
        for index, (left, right) in enumerate(
            zip(reference[0].lstm_layers, actual[0].lstm_layers, strict=True)
        ):
            state_errors[f"lstm{index}.h"] = error_metrics(backend, left.h, right.h)
            state_errors[f"lstm{index}.c"] = error_metrics(backend, left.c, right.c)
        finite = bool(np.isfinite(expected_loss) and np.isfinite(actual_loss))
        row_passed = finite and abs(expected_loss - actual_loss) <= FORWARD_CEILING
        row_passed &= all(
            _within(value, GRADIENT_CEILING) for value in param_errors.values()
        )
        row_passed &= all(
            _within(value, FORWARD_CEILING) for value in state_errors.values()
        )
        passed &= row_passed
        rows.append(
            {
                "update": update,
                "reference_loss": expected_loss,
                "production_loss": actual_loss,
                "perplexity": float(np.exp(actual_loss)),
                "parameter_errors": param_errors,
                "state_errors": state_errors,
                "finite": finite,
                "passed": bool(row_passed),
            }
        )
    return {"updates": rows, "passed": bool(passed)}


def reproducibility(backend, *, implementation: str = "phase1") -> dict[str, object]:
    corpus = backend.xp.asarray(load_ptb()["train"], dtype=backend.xp.int64)
    first = _model(backend, implementation, 424242)
    second = _model(backend, implementation, 424242)
    losses = [[], []]
    for update in range(5):
        xs, targets = _batch(backend, corpus, update)
        losses[0].append(_update(backend, first, xs, targets, 7000 + update))
        losses[1].append(_update(backend, second, xs, targets, 7000 + update))
    errors = {
        name: error_metrics(
            backend, parameter.data, dict(second[0].named_parameters())[name].data
        )
        for name, parameter in first[0].named_parameters()
    }
    for index, (left, right) in enumerate(
        zip(first[0].lstm_layers, second[0].lstm_layers, strict=True)
    ):
        errors[f"state.lstm{index}.h"] = error_metrics(backend, left.h, right.h)
        errors[f"state.lstm{index}.c"] = error_metrics(backend, left.c, right.c)
    loss_errors = [
        abs(left - right) for left, right in zip(losses[0], losses[1], strict=True)
    ]
    bitwise_loss_match = losses[0] == losses[1]
    passed = max(loss_errors, default=0.0) <= FORWARD_CEILING
    passed &= all(_within(value, GRADIENT_CEILING) for value in errors.values())
    return {
        "losses": losses,
        "loss_max_absolute": max(loss_errors, default=0.0),
        "bitwise_loss_match": bitwise_loss_match,
        "errors": errors,
        "passed": bool(passed),
    }
