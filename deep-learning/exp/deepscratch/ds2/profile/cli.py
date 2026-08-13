"""Volume-owned profiling dispatch for DeepScratch DS2."""

from __future__ import annotations

from pathlib import Path

from exp.framework.execution.parsing import parse_experiment_ids


def profile(
    *,
    experiment: list[str] | None = None,
    device: list[str] | None = None,
    mode: str = "all",
    output_dir: Path | None = None,
    vsweap: bool = False,
    vocab_size: list[int] | None = None,
    condition: list[str] | None = None,
    update_warmup: int = 20,
    update_repetitions: int = 5,
    measured_updates: int = 50,
    vsweap_timing: str = "window",
    reverse_vocab_order: bool = False,
) -> None:
    selected = parse_experiment_ids(experiment or [])
    if selected == ["e02"]:
        from .e02.api import DEFAULT_RESULTS, run

        if vocab_size and not vsweap:
            raise ValueError("--vocab-size requires --vsweap")
        if vsweap:
            from .e02.vsweap import run as run_vsweap

            run_vsweap(
                devices=tuple(device or ("cuda:0",)),
                conditions=tuple(condition) if condition else None,
                vocab_sizes=tuple(vocab_size) if vocab_size else None,
                warmup_updates=update_warmup,
                measured_updates=measured_updates,
                repetitions=update_repetitions,
                timing_source=vsweap_timing,
                reverse_vocab_order=reverse_vocab_order,
                output_dir=output_dir or DEFAULT_RESULTS,
            )
            return

        run(
            devices=tuple(device or ("cpu", "cuda:0")),
            conditions=tuple(condition) if condition else None,
            mode=mode,
            update_warmup=update_warmup,
            update_repetitions=update_repetitions,
            measured_updates=measured_updates,
            output_dir=output_dir or DEFAULT_RESULTS,
        )
        return
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
    raise ValueError("DS2 profiling requires exactly -e 02, -e 05, or -e 06")
