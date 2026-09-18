"""File serialization helpers for JSON and CSV run records."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def write_metric_rows_csv(
    path: Path, *, run_key: str, rows: list[tuple[int, str, float]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["run_key", "step", "metric", "value", "timestamp"])
        for row in rows:
            writer.writerow([run_key, *row, time.time()])


def write_runtime_history_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "step_type",
                "step",
                "train_s",
                "eval_s",
                "checkpoint_s",
                "throughput_samples_per_s",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_memory_history_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "timestamp_s",
                "cpu_rss_bytes",
                "gpu_used_bytes",
                "gpu_reserved_bytes",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


__all__ = [
    "write_json",
    "write_memory_history_csv",
    "write_metric_rows_csv",
    "write_runtime_history_csv",
    "write_text",
]
