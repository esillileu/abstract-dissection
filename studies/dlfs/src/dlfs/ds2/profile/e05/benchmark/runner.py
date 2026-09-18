"""Runner orchestrating BetterRnnlm Phase benchmark suite."""

from __future__ import annotations

import json
from pathlib import Path

from deepscratch.core import BackendConfig, make_backend

from dlfs.ds2.profile.paths import profile_measurements

from .timelstm import benchmark_timelstm
from .timing import environment
from .workload import benchmark_full_update

DEFAULT_RESULTS = profile_measurements("e05")


def run(
    *,
    stage: str,
    device: str = "cuda:0",
    warmup: int = 20,
    iterations: int = 50,
    repetitions: int = 5,
    output_dir: Path = DEFAULT_RESULTS,
    timelstm_only: bool = False,
    profile: bool = False,
) -> Path:
    if stage not in {"baseline", "phase1", "phase2", "phase3"}:
        raise ValueError("stage must be baseline, phase1, phase2, or phase3")
    implementation = {
        "baseline": "reference",
        "phase1": "phase1",
        "phase2": "phase2",
        "phase3": "phase3",
    }[stage]
    backend = make_backend(
        BackendConfig(device=device, dtype="float32", seed=20260811, profile=profile)
    )
    result = {
        "schema_version": 1,
        "stage": stage,
        "implementation": implementation,
        "environment": environment(backend),
        "protocol": {
            "warmup": warmup,
            "iterations": iterations,
            "repetitions": repetitions,
        },
        "timelstm": benchmark_timelstm(
            backend,
            implementation=implementation,
            warmup=warmup,
            iterations=iterations,
            repetitions=repetitions,
        ),
    }
    if not timelstm_only:
        result["full_update"] = benchmark_full_update(
            backend,
            implementation=implementation,
            warmup=warmup,
            iterations=iterations,
            repetitions=repetitions,
            profile=profile,
        )
    stage_dir = output_dir / stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    path = stage_dir / "benchmark.json"
    if stage == "baseline" and path.exists():
        raise FileExistsError(f"refusing to overwrite immutable baseline: {path}")
    path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(path)
    return path
