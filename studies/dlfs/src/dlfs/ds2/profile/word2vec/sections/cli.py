"""CLI command handler for Word2Vec section and module profiling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..workloads import (
    CONDITIONS,
    _load_data,
    _metadata,
)
from .runner import COMPONENTS, DEFAULT_OUTPUT, profile_modules


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Expose independently measurable Word2Vec workload sections."
    )
    parser.add_argument(
        "--condition",
        action="append",
        choices=CONDITIONS,
        dest="conditions",
    )
    parser.add_argument(
        "--component",
        action="append",
        choices=COMPONENTS,
        dest="components",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--warmup-iterations", type=int, default=5)
    parser.add_argument("--measured-iterations", type=int, default=20)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    if min(args.batch_size, args.measured_iterations) < 1 or args.warmup_iterations < 0:
        parser.error(
            "batch size and measured iterations must be positive; warmup must "
            "be non-negative"
        )

    backend, corpus, contexts, targets = _load_data(args.device)
    rows = []
    for condition in args.conditions or CONDITIONS:
        backend.seed(1)
        np.random.seed(1)
        condition_rows = profile_modules(
            condition,
            corpus=corpus,
            contexts=contexts,
            targets=targets,
            backend=backend,
            batch_size=args.batch_size,
            components=(
                tuple(args.components) if args.components is not None else None
            ),
            warmup_iterations=args.warmup_iterations,
            measured_iterations=args.measured_iterations,
        )
        rows.extend(condition_rows)
        for row in condition_rows:
            timing = row["timing"]
            assert isinstance(timing, dict)
            print(
                f"{condition} {row['component']}: "
                f"{float(timing['mean_ms']):.3f} ± "
                f"{float(timing['stdev_ms']):.3f} ms",
                flush=True,
            )

    payload = {
        "schema_version": 1,
        "metadata": _metadata(backend, stage="modules"),
        "results": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"saved: {args.output}", flush=True)
