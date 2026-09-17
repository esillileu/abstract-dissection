"""Gate runner orchestrating TimeLSTM correctness validation."""

from __future__ import annotations

import json
from pathlib import Path

from deepscratch.core import BackendConfig, make_backend

from ..benchmark import DEFAULT_RESULTS
from .layers import compare_lstm
from .metrics import (
    FORWARD_CEILING,
    GRADIENT_CEILING,
    TOLERANCE_GRID,
    _selected_tolerance,
)
from .models import lockstep, reproducibility


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
    cuda_backend = make_backend(
        BackendConfig(device=devices[-1], dtype="float32", seed=0)
    )
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
        all(
            case["passed"]
            for device in comparisons.values()
            for case in device.values()
        )
        and lockstep_result["passed"]
        and reproducibility_result["passed"]
        and result["tolerance_selection"]["forward_selected"] is not None
        and result["tolerance_selection"]["gradient_selected"] is not None
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / stage / "correctness.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(path)
    if not result["passed"]:
        raise SystemExit("Phase 1 correctness gate failed")
    return path
