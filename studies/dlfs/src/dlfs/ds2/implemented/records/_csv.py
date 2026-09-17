"""Low-level CSV serialization helpers and source-curve metric name mapping.

These functions are stateless and can be used independently of DS2Records.
"""

from __future__ import annotations

import csv
from pathlib import Path


def materialize_scalars(entries: list[tuple[dict[str, object], str]]) -> None:
    """Batch-convert Tensor values in-place using the backend's to_numpy."""
    groups: dict[int, tuple[object, list[tuple[dict[str, object], str, object]]]] = {}
    for row, key in entries:
        value = row.get(key)
        if not hasattr(value, "backend") or not hasattr(value, "data"):
            continue
        group = groups.setdefault(id(value.backend), (value.backend, []))
        group[1].append((row, key, value))
    for backend, values in groups.values():
        stacked = backend.xp.stack([value.data.reshape(()) for _, _, value in values])
        host_values = backend.to_numpy(stacked)
        for (row, key, _), host_value in zip(values, host_values, strict=True):
            row[key] = float(host_value)


def csv_value(value: object) -> object:
    """Convert a potentially-Tensor value to a plain Python scalar."""
    if hasattr(value, "backend") and hasattr(value, "data"):
        return value.backend.scalar_to_float(value.data)
    return value


def append_csv(
    name: str,
    path: Path,
    rows: list[dict[str, object]],
    *,
    columns: list[str],
    written_rows: dict[str, int],
) -> None:
    """Append new rows to a CSV file, writing the header on first write.

    Args:
        name: Key used to track how many rows have already been written.
        path: Destination CSV file path.
        rows: Full list of rows accumulated so far.
        columns: CSV fieldnames (and row ordering).
        written_rows: Mutable dict tracking per-file write progress (modified
            in-place by this function).
    """
    initialized = name in written_rows
    start = written_rows.get(name, 0)
    if not initialized or not path.exists():
        start = 0
    pending = rows[start:]
    if initialized and not pending and path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if initialized and path.exists() else "w"
    with path.open(mode, newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        if mode == "w":
            writer.writeheader()
        for row in pending:
            writer.writerow({key: csv_value(row.get(key, "")) for key in columns})
    written_rows[name] = len(rows)


def source_curve_metric_name(metric: str) -> str | None:
    """Map a source-curve metric tag to its MLflow metric key."""
    if metric == "loss":
        return "series/train/loss"
    if metric == "book_loss":
        return "series/train/book_loss"
    if metric == "perplexity":
        return "series/train/perplexity"
    if metric == "exact_match_accuracy":
        return "series/eval_test/exact_match_accuracy"
    return None
