"""Lockstep execution runner for e02 fused Word2Vec validation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from deepscratch.core import BackendConfig, Tensor, make_backend
from deepscratch.nn.model.architecture import (
    CBOW,
    CBOWBatchAdapter,
    FusedNegativeSamplingCBOW,
    FusedNegativeSamplingSkipGram,
    SkipGram,
    SkipGramBatchAdapter,
)
from deepscratch.nn.objective import FusedNegativeSampling, NegativeSampling
from deepscratch.optim.SGD import Adam

from dlfs.ds2.profile.paths import profile_analysis

from .metrics import (
    GRADIENT_CEILING,
    LOSS_CEILING,
    PARAMETER_CEILING,
    _error_metrics,
    _optimizer_errors,
    _parameter_errors,
    _passed,
)
from .report import render_report

DEFAULT_OUTPUT = profile_analysis("e02") / "fused_validation.json"
DEFAULT_REPORT = profile_analysis("e02") / "fused_validation.md"


def _tensor(backend, value: np.ndarray) -> Tensor:
    return Tensor(backend.xp.asarray(value), backend=backend)


def _batches(kind: str) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return fixed batches; the third value contains only negative IDs."""
    rows = np.arange(20, dtype=np.int64).reshape(4, 5) % 8
    if kind == "cbow":
        return [
            (
                (rows + update)[:, :3] % 8,
                (rows[:, 0] + update + 1) % 8,
                (rows[:, 2:4] + update + 2) % 8,
            )
            for update in range(5)
        ]
    return [
        (
            (rows[:, 1:4] + update) % 8,
            (rows[:, 0] + update) % 8,
            np.stack(
                ((rows[:, 2:5] + update + 2) % 8, (rows[:, :3] + update + 3) % 8),
                axis=-1,
            ),
        )
        for update in range(5)
    ]


def _copy_parameters(source, target) -> None:
    source_params = dict(source.named_parameters())
    target_params = dict(target.named_parameters())
    if source_params.keys() != target_params.keys():
        raise AssertionError("ordinary and fused parameter names differ")
    for name in source_params:
        target_params[name].data[...] = source_params[name].data


def validate_kind(backend, kind: str) -> dict[str, object]:
    if kind == "cbow":
        ordinary_model = CBOW(8, 4, backend=backend)
        fused_model = FusedNegativeSamplingCBOW(8, 4, backend=backend)
        adapter = CBOWBatchAdapter()
    elif kind == "skipgram":
        ordinary_model = SkipGram(8, 4, backend=backend)
        fused_model = FusedNegativeSamplingSkipGram(8, 4, backend=backend)
        adapter = SkipGramBatchAdapter()
    else:
        raise ValueError(f"unsupported Word2Vec kind: {kind}")

    _copy_parameters(ordinary_model, fused_model)
    ordinary_objective = NegativeSampling(8, negative_samples=2, backend=backend)
    fused_objective = FusedNegativeSampling(8, negative_samples=2, backend=backend)
    ordinary_optimizer = Adam(list(ordinary_model.named_parameters()), lr=0.001)
    fused_optimizer = Adam(list(fused_model.named_parameters()), lr=0.001)
    rows = []

    for update, (first, second, negatives) in enumerate(_batches(kind), start=1):
        model_x, objective_t = adapter.prepare(
            _tensor(backend, first), _tensor(backend, second)
        )
        ordinary_batch = ordinary_objective.prepare(
            objective_t, replay_context=backend.xp.asarray(negatives)
        )
        fused_batch = fused_objective.prepare(
            objective_t, replay_context=backend.xp.asarray(negatives)
        )

        ordinary_prediction = ordinary_model.forward(
            model_x, candidates=ordinary_batch.candidates
        )
        ordinary_result = ordinary_objective.forward(
            ordinary_prediction,
            ordinary_batch.target,
            replay_context=ordinary_batch.replay_context,
            example_count=len(first),
        )
        ordinary_model.backward(ordinary_objective.backward())

        fused_result = fused_objective.forward_fused(
            fused_model,
            model_x,
            fused_batch,
            example_count=len(first),
        )
        fused_objective.backward_fused(fused_model)
        backend.synchronize()

        gradient_errors = {
            name: _error_metrics(
                backend, parameter.grad, dict(fused_model.named_parameters())[name].grad
            )
            for name, parameter in ordinary_model.named_parameters()
        }
        loss_error = _error_metrics(
            backend, ordinary_result.loss.data, fused_result.loss.data
        )
        ordinary_optimizer.update()
        fused_optimizer.update()
        backend.synchronize()
        parameter_errors = _parameter_errors(backend, ordinary_model, fused_model)
        optimizer_errors = _optimizer_errors(
            backend, ordinary_optimizer, fused_optimizer
        )
        row_passed = (
            loss_error["required_atol_rtol"] <= LOSS_CEILING
            and _passed(gradient_errors, GRADIENT_CEILING)
            and _passed(parameter_errors, PARAMETER_CEILING)
            and _passed(optimizer_errors, PARAMETER_CEILING)
        )
        rows.append(
            {
                "update": update,
                "ordinary_loss": float(
                    backend.scalar_to_float(ordinary_result.loss.data)
                ),
                "fused_loss": float(backend.scalar_to_float(fused_result.loss.data)),
                "loss_error": loss_error,
                "gradient_errors": gradient_errors,
                "parameter_errors": parameter_errors,
                "optimizer_state_errors": optimizer_errors,
                "passed": bool(row_passed),
            }
        )

    return {
        "kind": kind,
        "updates": rows,
        "passed": all(row["passed"] for row in rows),
    }


def run(
    output: Path = DEFAULT_OUTPUT,
    devices: tuple[str, ...] = ("cpu",),
    report: Path | None = None,
) -> dict[str, object]:
    comparisons: dict[str, object] = {}
    for device in devices:
        backend = make_backend(
            BackendConfig(device=device, dtype="float32", seed=20260821)
        )
        comparisons[device] = {
            kind: validate_kind(backend, kind) for kind in ("cbow", "skipgram")
        }
    result = {
        "schema_version": 1,
        "protocol": {
            "updates": 5,
            "dtype": "float32",
            "negative_samples": 2,
            "optimizer": "Adam(lr=0.001)",
            "same_negative_candidates": True,
            "loss_ceiling": LOSS_CEILING,
            "gradient_ceiling": GRADIENT_CEILING,
            "parameter_and_optimizer_ceiling": PARAMETER_CEILING,
        },
        "devices": list(devices),
        "comparisons": comparisons,
        "passed": all(
            case["passed"]
            for device in comparisons.values()
            for case in device.values()
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    report_path = report or output.with_suffix(".md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(result), encoding="utf-8")
    return result
