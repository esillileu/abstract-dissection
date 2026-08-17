"""Volume-owned profiling dispatch for DeepScratch DS2."""

from __future__ import annotations

from pathlib import Path

from exp.framework.execution.parsing import parse_experiment_ids


def profile(
    *,
    experiment: list[str] | None = None,
    device: list[str] | None = None,
    output_dir: Path | None = None,
    condition: list[str] | None = None,
    update_warmup: int = 20,
    update_repetitions: int = 5,
    measured_updates: int = 50,
) -> None:
    selected = parse_experiment_ids(experiment or [])
    if selected == ["e02"]:
        raise ValueError(
            "e02 profiling was promoted to the official e10/PF01 and "
            "e11/PF02 run paths"
        )
    if selected == ["e05"]:
        from .e05.benchmark import DEFAULT_RESULTS, run

        if len(device or ("cuda:0",)) != 1:
            raise ValueError("e05 profiling accepts exactly one --device")
        run(
            device=(device or ["cuda:0"])[0],
            warmup=update_warmup,
            iterations=measured_updates,
            repetitions=update_repetitions,
            output_dir=output_dir or DEFAULT_RESULTS,
        )
        return
    if selected == ["e06"]:
        from .e06.benchmark import DEFAULT_RESULTS, run

        if len(device or ("cuda:0",)) != 1:
            raise ValueError("e06 profiling accepts exactly one --device")
        run(
            device=(device or ["cuda:0"])[0],
            conditions=tuple(condition) if condition else tuple(),
            warmup=update_warmup,
            iterations=measured_updates,
            repetitions=update_repetitions,
            output_dir=output_dir or DEFAULT_RESULTS,
        )
        return
    raise ValueError("DS2 profiling requires exactly -e 05 or -e 06")
