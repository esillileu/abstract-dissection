from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from deepscratch.core import BackendConfig, make_backend

from repro_core.analysis.core import save_figure

from ..workloads import _metadata
from .constants import (
    CONDITIONS,
    CONTEXT_WIDTH,
    DEFAULT_RESULTS,
    EMBEDDING_SIZE,
    NEGATIVE_SAMPLES,
    _default_vocab_sizes,
    _validate_vocab_sizes,
)
from .crossover import _crossovers, _render_crossovers
from .measurement import _measure_condition
from .render import render_individual_scaling, render_scaling
from .workload import _synthetic_batches


def run(
    *,
    devices: tuple[str, ...] = ("cuda:0",),
    vocab_sizes: tuple[int, ...] | None = None,
    conditions: tuple[str, ...] | None = None,
    batch_size: int = 100,
    warmup_updates: int = 20,
    measured_updates: int = 50,
    repetitions: int = 5,
    timing_source: str = "window",
    reverse_vocab_order: bool = False,
    output_dir: Path = DEFAULT_RESULTS,
) -> None:
    """Measure synchronized update distributions at every scaling point."""
    if timing_source not in {"window", "event"}:
        raise ValueError("timing source must be 'window' or 'event'")
    selected_conditions = CONDITIONS if conditions is None else conditions
    unknown = set(selected_conditions) - set(CONDITIONS)
    if unknown:
        raise ValueError(
            "vocabulary-size scaling supports implemented conditions only: "
            f"{sorted(unknown)}"
        )
    if not selected_conditions:
        raise ValueError("vocabulary-size scaling requires at least one condition")
    if min(batch_size, measured_updates, repetitions) < 1 or warmup_updates < 0:
        raise ValueError(
            "batch size, measured updates, and repetitions must be positive; "
            "warmup updates must be non-negative"
        )
    if vocab_sizes is not None:
        _validate_vocab_sizes(vocab_sizes)

    for device in devices:
        device_vocab_sizes = (
            _default_vocab_sizes(device) if vocab_sizes is None else vocab_sizes
        )
        if reverse_vocab_order:
            device_vocab_sizes = tuple(reversed(device_vocab_sizes))
        backend = make_backend(
            BackendConfig(
                device=device,
                dtype="float32",
                seed=1,
                profile=device.startswith("cuda:"),
            )
        )
        rows: list[dict[str, object]] = []
        for vocab_size in device_vocab_sizes:
            contexts, targets = _synthetic_batches(
                vocab_size,
                batch_size=batch_size,
                update_count=(
                    1
                    + warmup_updates
                    + measured_updates
                    + measured_updates * repetitions
                ),
            )
            for condition in selected_conditions:
                backend.seed(1)
                np.random.seed(1)
                row = _measure_condition(
                    condition,
                    vocab_size=vocab_size,
                    contexts=contexts,
                    targets=targets,
                    backend=backend,
                    batch_size=batch_size,
                    warmup_updates=warmup_updates,
                    measured_updates=measured_updates,
                    repetitions=repetitions,
                )
                rows.append(row)
                if row["status"] == "ok":
                    print(
                        f"[{device}] V={vocab_size:,} {condition}: "
                        f"{float(row['update_ms']):.3f} ms/update",
                        flush=True,
                    )
                else:
                    print(
                        f"[{device}] V={vocab_size:,} {condition}: "
                        f"{row['status']} ({row['error']})",
                        flush=True,
                    )

        device_dir = output_dir / device.replace(":", "")
        device_dir.mkdir(parents=True, exist_ok=True)
        output = device_dir / "vocabulary_size_scaling.json"
        payload = {
            "schema_version": 3,
            "metadata": {
                **_metadata(backend, stage="vocabulary_size_scaling"),
                "method": (
                    "synthetic uniform vocabulary; current implemented update "
                    "path including dense Adam and post-update loss; device "
                    "synchronization at cold, warmup, event-distribution, and "
                    "repeated throughput-window boundaries"
                ),
                "embedding_size": EMBEDDING_SIZE,
                "context_width": CONTEXT_WIDTH,
                "negative_samples": NEGATIVE_SAMPLES,
                "batch_size": batch_size,
                "warmup_updates": warmup_updates,
                "measured_updates": measured_updates,
                "repetitions": repetitions,
                "vocab_sizes": list(device_vocab_sizes),
                "vocab_order": "descending" if reverse_vocab_order else "ascending",
                "timing_source": timing_source,
            },
            "results": rows,
            "crossovers": _crossovers(rows),
        }
        output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"saved: {output}", flush=True)
        figure = render_scaling(payload)
        figure_output = save_figure(figure, device_dir / "vocabulary_size_scaling.png")
        plt.close(figure)
        print(f"saved: {figure_output}", flush=True)
        for model, figure in render_individual_scaling(payload):
            individual_output = save_figure(
                figure,
                device_dir / f"vocabulary_size_scaling-{model.lower()}.png",
            )
            plt.close(figure)
            print(f"saved: {individual_output}", flush=True)
        print(_render_crossovers(device, payload["crossovers"]), flush=True)
