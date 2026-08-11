"""Correctness and determinism gates for the TimeLSTM Phase 1 change."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mlprosection import Tensor
from mlprosection.core.backend import BackendConfig, make_backend
from mlprosection.datasets import load_ptb
from mlprosection.nn.layers import TimeLSTM
from mlprosection.nn.model.architecture.recurrent import BetterRnnlm
from mlprosection.optim.SGD import SGD
from mlprosection.optim.transform import ClipGradNorm

from .benchmark import DEFAULT_RESULTS
from .reference import ReferenceTimeLSTM, replace_better_rnnlm_lstms
from .phase1 import Phase1TimeLSTM, replace_better_rnnlm_lstms as replace_phase1_lstms
from .phase2 import Phase2TimeLSTM, replace_better_rnnlm_lstms as replace_phase2_lstms
from .phase3 import (
    Phase3TemporalSoftmaxCrossEntropy,
    UnfusedTemporalSoftmaxCrossEntropy,
)


FORWARD_CEILING = 1e-4
GRADIENT_CEILING = 1e-3
TOLERANCE_GRID = (1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3)


def _host(backend, value) -> np.ndarray:
    return np.asarray(backend.to_numpy(value))


def error_metrics(backend, reference, actual) -> dict[str, float]:
    left, right = _host(backend, reference), _host(backend, actual)
    absolute = np.abs(left.astype(np.float64) - right.astype(np.float64))
    scale = np.maximum(np.maximum(np.abs(left), np.abs(right)), 1e-6)
    return {
        "max_absolute": float(absolute.max(initial=0.0)),
        "max_relative": float((absolute / scale).max(initial=0.0)),
        "required_atol_rtol": float((absolute / (1.0 + scale)).max(initial=0.0)),
    }


def _within(metrics: dict[str, float], ceiling: float) -> bool:
    # This is the standard combined atol/rtol test expressed using the two
    # recorded maxima.  The raw maxima remain in the artifact for inspection.
    return metrics["required_atol_rtol"] <= ceiling


def _copy_lstm(source, target) -> None:
    target.Wx.data[...] = source.Wx.data
    target.Wh.data[...] = source.Wh.data
    target.b.data[...] = source.b.data


def compare_lstm(
    backend,
    shape: tuple[int, int, int, int],
    seed: int,
    *,
    implementation: str = "phase1",
):
    n, time_size, input_size, hidden_size = shape
    xp = backend.xp
    xp.random.seed(seed)
    reference = ReferenceTimeLSTM(
        input_size, hidden_size, stateful=True, backend=backend
    )
    actual_cls = {
        "phase1": Phase1TimeLSTM,
        "phase2": Phase2TimeLSTM,
        "phase3": Phase2TimeLSTM,
    }.get(implementation, TimeLSTM)
    actual = actual_cls(input_size, hidden_size, stateful=True, backend=backend)
    _copy_lstm(reference, actual)
    xs = Tensor(xp.random.randn(n, time_size, input_size).astype(xp.float32), backend=backend)
    dhs = Tensor(xp.random.randn(n, time_size, hidden_size).astype(xp.float32), backend=backend)
    h0 = xp.random.randn(n, hidden_size).astype(xp.float32)
    c0 = xp.random.randn(n, hidden_size).astype(xp.float32)
    reference.set_state(h0, c0)
    actual.set_state(h0, c0)
    expected = reference.forward(xs)
    observed = actual.forward(xs)
    outputs = {
        "output": error_metrics(backend, expected.data, observed.data),
        "final_h": error_metrics(backend, reference.h, actual.h),
        "final_c": error_metrics(backend, reference.c, actual.c),
    }
    expected_dx = reference.backward(dhs)
    observed_dx = actual.backward(dhs)
    gradients = {
        "dx": error_metrics(backend, expected_dx.data, observed_dx.data),
        "dWx": error_metrics(backend, reference.Wx.grad, actual.Wx.grad),
        "dWh": error_metrics(backend, reference.Wh.grad, actual.Wh.grad),
        "db": error_metrics(backend, reference.b.grad, actual.b.grad),
        "dh": error_metrics(backend, reference.dh.data, actual.dh.data),
    }

    # Consecutive stateful forward and lifecycle contracts.
    next_x = Tensor(xp.random.randn(n, 2, input_size).astype(xp.float32), backend=backend)
    outputs["stateful_next"] = error_metrics(
        backend, reference.forward(next_x).data, actual.forward(next_x).data
    )
    reference.detach_state()
    actual.detach_state()
    assert reference.layers == [] and actual.layers == []
    reference.reset_state()
    actual.reset_state()
    assert reference.h is None and actual.h is None and reference.c is None and actual.c is None
    no_cache = actual.forward(next_x, cache=False)
    assert no_cache.shape == (n, 2, hidden_size) and actual.layers == []
    try:
        actual.backward(Tensor(xp.zeros_like(no_cache.data), backend=backend))
    except RuntimeError:
        pass
    else:
        raise AssertionError("cache=False unexpectedly allowed backward")
    changed = Tensor(xp.random.randn(n + 1, 1, input_size).astype(xp.float32), backend=backend)
    actual.forward(changed, cache=False)
    assert actual.h.shape[0] == n + 1 and actual.c.shape[0] == n + 1

    passed = all(_within(value, FORWARD_CEILING) for value in outputs.values())
    passed &= all(_within(value, GRADIENT_CEILING) for value in gradients.values())
    return {"outputs": outputs, "gradients": gradients, "passed": bool(passed)}


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
    positions = (offsets[:, None] + update * time_size + xp.arange(time_size)[None, :]) % size
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
        row_passed &= all(_within(value, GRADIENT_CEILING) for value in param_errors.values())
        row_passed &= all(_within(value, FORWARD_CEILING) for value in state_errors.values())
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
        name: error_metrics(backend, parameter.data, dict(second[0].named_parameters())[name].data)
        for name, parameter in first[0].named_parameters()
    }
    for index, (left, right) in enumerate(zip(first[0].lstm_layers, second[0].lstm_layers, strict=True)):
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


def _selected_tolerance(maximum: float, ceiling: float) -> float | None:
    target = 2 * maximum
    return next((value for value in TOLERANCE_GRID if value >= target and value <= ceiling), None)


def run(output_dir: Path = DEFAULT_RESULTS, *, stage: str = "phase1") -> Path:
    if stage not in {"phase1", "phase2", "phase3"}:
        raise ValueError("validation stage must be phase1, phase2, or phase3")
    devices = ["cpu"]
    try:
        make_backend(BackendConfig(device="cuda:0", dtype="float32"))
    except RuntimeError:
        pass
    else:
        devices.append("cuda:0")
    shapes = ((1, 1, 1, 1), (2, 3, 4, 5), (3, 2, 5, 4), (20, 35, 650, 650))
    implementation = stage
    comparisons = {}
    for device in devices:
        backend = make_backend(BackendConfig(device=device, dtype="float32", seed=0))
        comparisons[device] = {
            f"seed={seed},shape={shape}": compare_lstm(
                backend, shape, seed, implementation=implementation
            )
            for seed in (1, 7, 23)
            for shape in shapes
        }
    cuda_backend = make_backend(BackendConfig(device=devices[-1], dtype="float32", seed=0))
    lockstep_result = lockstep(cuda_backend, implementation=implementation)
    reproducibility_result = reproducibility(
        cuda_backend, implementation=implementation
    )
    output_errors = [
        metric
        for device in comparisons.values()
        for case in device.values()
        for metric in case["outputs"].values()
    ]
    gradient_errors = [
        metric
        for device in comparisons.values()
        for case in device.values()
        for metric in case["gradients"].values()
    ]
    max_forward = max(value["required_atol_rtol"] for value in output_errors)
    max_gradient = max(value["required_atol_rtol"] for value in gradient_errors)
    result = {
        "schema_version": 1,
        "devices": devices,
        "comparisons": comparisons,
        "lockstep": lockstep_result,
        "reproducibility": reproducibility_result,
        "tolerance_selection": {
            "grid": TOLERANCE_GRID,
            "forward_observed_max": max_forward,
            "gradient_observed_max": max_gradient,
            "forward_selected": _selected_tolerance(max_forward, FORWARD_CEILING),
            "gradient_selected": _selected_tolerance(max_gradient, GRADIENT_CEILING),
            "forward_hard_ceiling": FORWARD_CEILING,
            "gradient_hard_ceiling": GRADIENT_CEILING,
        },
    }
    result["passed"] = (
        all(case["passed"] for device in comparisons.values() for case in device.values())
        and lockstep_result["passed"]
        and reproducibility_result["passed"]
        and result["tolerance_selection"]["forward_selected"] is not None
        and result["tolerance_selection"]["gradient_selected"] is not None
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / stage / "correctness.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(path)
    if not result["passed"]:
        raise SystemExit("Phase 1 correctness gate failed")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument(
        "--stage", choices=("phase1", "phase2", "phase3"), default="phase1"
    )
    arguments = parser.parse_args()
    run(arguments.output_dir, stage=arguments.stage)
