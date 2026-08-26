"""Numerical lockstep validation for the e02 fused Word2Vec paths.

The ordinary and fused objectives receive the same negative candidates.  This
is intentional: it isolates the fused score/loss/gradient implementation from
the sampler's RNG stream and lets us compare the Adam trajectory update by
update.
"""

from __future__ import annotations

import argparse
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

DEFAULT_OUTPUT = Path("exp/deepscratch/ds2/profile/e02/results/fused_validation.json")
DEFAULT_REPORT = Path("exp/deepscratch/ds2/profile/e02/results/fused_validation.md")
LOSS_CEILING = 1e-5
GRADIENT_CEILING = 1e-4
PARAMETER_CEILING = 1e-4


def _error_metrics(backend, expected, actual) -> dict[str, float]:
    left = np.asarray(backend.to_numpy(expected), dtype=np.float64)
    right = np.asarray(backend.to_numpy(actual), dtype=np.float64)
    absolute = np.abs(left - right)
    scale = np.maximum(np.maximum(np.abs(left), np.abs(right)), 1e-6)
    return {
        "max_absolute": float(absolute.max(initial=0.0)),
        "max_relative": float((absolute / scale).max(initial=0.0)),
        "required_atol_rtol": float((absolute / (1.0 + scale)).max(initial=0.0)),
    }


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


def _parameter_errors(backend, source, target) -> dict[str, dict[str, float]]:
    other = dict(target.named_parameters())
    return {
        name: _error_metrics(backend, parameter.data, other[name].data)
        for name, parameter in source.named_parameters()
    }


def _optimizer_errors(
    backend, source: Adam, target: Adam
) -> dict[str, dict[str, float]]:
    return {
        f"m.{name}": _error_metrics(backend, source.m[name], target.m[name])
        for name in source.m
    } | {
        f"v.{name}": _error_metrics(backend, source.v[name], target.v[name])
        for name in source.v
    }


def _passed(metrics: dict[str, dict[str, float]], ceiling: float) -> bool:
    return all(value["required_atol_rtol"] <= ceiling for value in metrics.values())


def _max_error(rows: list[dict[str, object]], group: str, field: str) -> float:
    return max(
        float(metric[field])
        for row in rows
        for metric in (row[group].values() if group != "loss_error" else (row[group],))
    )


def render_report(result: dict[str, object]) -> str:
    protocol = result["protocol"]
    lines = [
        "# e02 fused negative-sampling validation",
        "",
        "## Result",
        "",
        "The ordinary and fused Word2Vec paths were compared in lockstep.",
        "Both paths used the same initial parameters, batches, negative",
        "candidate IDs, and Adam optimizer state.",
        "",
        "| device | model | updates | max loss abs. error | max gradient combined error | max parameter combined error | max Adam-state combined error | status |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for device, cases in result["comparisons"].items():
        for kind, case in cases.items():
            rows = case["updates"]
            lines.append(
                "| {device} | {kind} | {updates} | {loss:.3e} | {gradient:.3e} | "
                "{parameter:.3e} | {state:.3e} | {status} |".format(
                    device=device,
                    kind=kind,
                    updates=len(rows),
                    loss=_max_error(rows, "loss_error", "max_absolute"),
                    gradient=_max_error(rows, "gradient_errors", "required_atol_rtol"),
                    parameter=_max_error(
                        rows, "parameter_errors", "required_atol_rtol"
                    ),
                    state=_max_error(
                        rows, "optimizer_state_errors", "required_atol_rtol"
                    ),
                    status="PASS" if case["passed"] else "FAIL",
                )
            )
    lines.extend(
        [
            "",
            "## Protocol",
            "",
            f"- dtype: `{protocol['dtype']}`",
            f"- updates per model: `{protocol['updates']}`",
            f"- negative samples: `{protocol['negative_samples']}`",
            f"- optimizer: `{protocol['optimizer']}`",
            f"- same negative candidates: `{protocol['same_negative_candidates']}`",
            f"- loss ceiling: `{protocol['loss_ceiling']}`",
            f"- gradient ceiling: `{protocol['gradient_ceiling']}`",
            f"- parameter/optimizer-state ceiling: `{protocol['parameter_and_optimizer_ceiling']}`",
            "- CUDA comparisons synchronize before reading gradients and parameters.",
            "",
            f"Overall status: **{'PASS' if result['passed'] else 'FAIL'}**",
            "",
            "The complete per-update values are stored in the adjacent JSON artifact.",
            "",
        ]
    )
    return "\n".join(lines)


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--device", action="append", default=None)
    args = parser.parse_args()
    devices = tuple(args.device or ("cpu",))
    result = run(args.output, devices=devices, report=args.report)
    print(json.dumps(result, indent=2))
    print(f"JSON report: {args.output}")
    print(f"Markdown report: {args.report or args.output.with_suffix('.md')}")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
