from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from deepscratch.profiling import (
    BenchmarkRunner,
    SectionRecorder,
    estimate_training_time,
)

from .condition import _batch, _build_condition, _run_updates
from .constants import CONDITIONS, DEFAULT_EPOCHS, DEFAULT_OUTPUT, STAGES
from .env import _load_data, _metadata
from .result import ConditionResult


def profile_condition(
    condition: str,
    *,
    corpus,
    contexts,
    targets,
    backend,
    batch_size: int,
    epochs: int,
    warmup_updates: int,
    measured_updates: int,
    phase_updates: int,
    repetitions: int,
) -> ConditionResult:
    workload, model_name, objective_name, implementation = _build_condition(
        condition,
        corpus=corpus,
        contexts=contexts,
        targets=targets,
        backend=backend,
    )
    next_index = 0

    def update_once() -> None:
        nonlocal next_index
        batch_x, batch_t = _batch(workload, next_index, batch_size)
        workload.update(batch_x, batch_t)
        next_index += 1

    benchmark = BenchmarkRunner(backend).measure_update_protocol(
        f"{condition}.update",
        update_once,
        warmup_iterations=warmup_updates,
        measured_iterations=measured_updates,
        repetitions=repetitions,
    )

    recorder = SectionRecorder(backend)
    if phase_updates:
        _run_updates(
            workload,
            start_index=next_index,
            updates=phase_updates,
            batch_size=batch_size,
            recorder=recorder,
        )
    phase_timings = recorder.stats()
    phase_means = {name: timing.mean_ms for name, timing in phase_timings.items()}
    phase_total = sum(phase_means.values())
    estimate = estimate_training_time(
        benchmark.timing,
        dataset_samples=len(contexts),
        batch_size=batch_size,
        epochs=epochs,
        cold_update_ms=benchmark.cold_ms,
    )
    first_epoch_seconds = (
        benchmark.cold_ms
        + benchmark.timing.mean_ms * max(estimate.updates_per_epoch - 1, 0)
    ) / 1_000
    return ConditionResult(
        condition=condition,
        implementation=implementation,
        model=model_name,
        objective=objective_name,
        batch_size=batch_size,
        dataset_samples=len(contexts),
        updates_per_epoch=estimate.updates_per_epoch,
        estimated_epochs=epochs,
        warmup_updates=warmup_updates,
        measured_updates=measured_updates,
        repetitions=repetitions,
        cold_ms_per_update=benchmark.cold_ms,
        warmup_total_ms=benchmark.warmup_total_ms,
        warmup_mean_ms=benchmark.warmup_mean_ms,
        steady_event_mean_ms_per_update=benchmark.event_timing.mean_ms,
        steady_event_stdev_ms_per_update=benchmark.event_timing.stdev_ms,
        steady_event_min_ms_per_update=benchmark.event_timing.min_ms,
        steady_event_max_ms_per_update=benchmark.event_timing.max_ms,
        steady_event_p50_ms_per_update=benchmark.event_timing.p50_ms,
        steady_event_p95_ms_per_update=benchmark.event_timing.p95_ms,
        mean_ms_per_update=benchmark.timing.mean_ms,
        stdev_ms_per_update=benchmark.timing.stdev_ms,
        min_ms_per_update=benchmark.timing.min_ms,
        max_ms_per_update=benchmark.timing.max_ms,
        samples_per_second=(batch_size / (benchmark.timing.mean_ms / 1_000)),
        estimated_first_epoch_seconds=first_epoch_seconds,
        estimated_seconds_per_epoch=estimate.mean_seconds_per_epoch,
        estimated_repeat_stdev_seconds_per_epoch=(
            estimate.repeat_stdev_seconds_per_epoch
        ),
        estimated_seconds_total=estimate.mean_seconds_total,
        estimated_repeat_stdev_seconds_total=estimate.repeat_stdev_seconds_total,
        phase_ms_per_update=phase_means,
        phase_stats={name: asdict(timing) for name, timing in phase_timings.items()},
        phase_share={name: value / phase_total for name, value in phase_means.items()}
        if phase_total
        else {},
    )


def _stage_value(
    explicit: int | None,
    *,
    stage: str,
    name: str,
) -> int:
    return int(STAGES[stage][name]) if explicit is None else explicit


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--condition",
        action="append",
        choices=CONDITIONS,
        dest="conditions",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Execution device: cpu or cuda:N (default: cuda:0).",
    )
    parser.add_argument(
        "--stage",
        choices=tuple(STAGES),
        default="estimate",
        help="update=quick precise estimate, estimate=stable estimate, detail=phases.",
    )
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--warmup-updates", type=int)
    parser.add_argument("--measured-updates", type=int)
    parser.add_argument("--phase-updates", type=int)
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    warmup_updates = _stage_value(
        args.warmup_updates, stage=args.stage, name="warmup_updates"
    )
    measured_updates = _stage_value(
        args.measured_updates, stage=args.stage, name="measured_updates"
    )
    phase_updates = _stage_value(
        args.phase_updates, stage=args.stage, name="phase_updates"
    )
    repetitions = _stage_value(args.repetitions, stage=args.stage, name="repetitions")
    if (
        min(
            args.batch_size,
            args.epochs,
            measured_updates,
            repetitions,
        )
        < 1
        or min(warmup_updates, phase_updates) < 0
    ):
        parser.error(
            "batch size, epochs, measured updates, and repetitions must be "
            "positive; warmup and phase updates must be non-negative"
        )

    backend, corpus, contexts, targets = _load_data(args.device)
    results = []
    for condition in args.conditions or CONDITIONS:
        backend.seed(1)
        np.random.seed(1)
        result = profile_condition(
            condition,
            corpus=corpus,
            contexts=contexts,
            targets=targets,
            backend=backend,
            batch_size=args.batch_size,
            epochs=args.epochs,
            warmup_updates=warmup_updates,
            measured_updates=measured_updates,
            phase_updates=phase_updates,
            repetitions=repetitions,
        )
        results.append(result)
        print(
            f"{condition}: {result.mean_ms_per_update:.3f} ± "
            f"{result.stdev_ms_per_update:.3f} ms/update, "
            f"{result.estimated_seconds_per_epoch:.1f} ± "
            f"{result.estimated_repeat_stdev_seconds_per_epoch:.1f} s/epoch, "
            f"{result.estimated_seconds_total:.1f} ± "
            f"{result.estimated_repeat_stdev_seconds_total:.1f} "
            f"s/{args.epochs} epochs",
            flush=True,
        )
    payload = {
        "schema_version": 6,
        "metadata": _metadata(backend, stage=args.stage),
        "results": [asdict(result) for result in results],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"saved: {args.output}", flush=True)


if __name__ == "__main__":
    main()
