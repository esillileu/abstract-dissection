"""Kernel extraction and classification queries for Nsight Systems traces."""

from __future__ import annotations

import itertools
import sqlite3


def _has_table(connection: sqlite3.Connection, table: str) -> bool:
    return (
        connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone()
        is not None
    )


def _kernel_category(name: str) -> str:
    lowered = name.lower()
    if "gemm" in lowered or "cublas" in lowered:
        return "gemm"
    if "time_lstm_forward_f32" in lowered or "time_lstm_backward_f32" in lowered:
        return "elementwise"
    if lowered.startswith("cupy_") and "sum" not in lowered:
        return "elementwise"
    if "sum" in lowered or "reduce" in lowered:
        return "reduction"
    return "other"


def _kernels_in_ranges(
    connection: sqlite3.Connection, pattern: str
) -> list[tuple[str, str, int, int, int]]:
    return connection.execute(
        """
        SELECT ranges.text, names.value, kernels.start, kernels.end,
               kernels.streamId
        FROM CUPTI_ACTIVITY_KIND_KERNEL AS kernels
        JOIN StringIds AS names ON names.id = kernels.shortName
        JOIN CUPTI_ACTIVITY_KIND_RUNTIME AS runtime
          ON runtime.correlationId = kernels.correlationId
        JOIN NVTX_EVENTS AS ranges
          ON ranges.globalTid = runtime.globalTid
         AND runtime.start BETWEEN ranges.start AND ranges.end
        WHERE ranges.text LIKE ? AND ranges.end IS NOT NULL
        """,
        (pattern,),
    ).fetchall()


def _kernels_in_nested_ranges(
    connection: sqlite3.Connection, inner_pattern: str, outer_pattern: str
) -> list[tuple[str, str, int, int, int]]:
    return connection.execute(
        """
        SELECT inner_ranges.text, names.value, kernels.start, kernels.end,
               kernels.streamId
        FROM CUPTI_ACTIVITY_KIND_KERNEL AS kernels
        JOIN StringIds AS names ON names.id = kernels.shortName
        JOIN CUPTI_ACTIVITY_KIND_RUNTIME AS runtime
          ON runtime.correlationId = kernels.correlationId
        JOIN NVTX_EVENTS AS inner_ranges
          ON inner_ranges.globalTid = runtime.globalTid
         AND runtime.start BETWEEN inner_ranges.start AND inner_ranges.end
        JOIN NVTX_EVENTS AS outer_ranges
          ON outer_ranges.globalTid = inner_ranges.globalTid
         AND inner_ranges.start BETWEEN outer_ranges.start AND outer_ranges.end
        WHERE inner_ranges.text LIKE ? AND outer_ranges.text LIKE ?
          AND inner_ranges.end IS NOT NULL AND outer_ranges.end IS NOT NULL
        """,
        (inner_pattern, outer_pattern),
    ).fetchall()


def _kernel_summary(rows: list[tuple[str, str, int, int, int]]) -> dict[str, object]:
    categories: dict[str, dict[str, float | int]] = {}
    for _range, name, start, end, _stream in rows:
        category = _kernel_category(name)
        values = categories.setdefault(category, {"count": 0, "time_ms": 0.0})
        values["count"] += 1
        values["time_ms"] += (end - start) / 1_000_000
    gaps_ns = 0
    for stream in {row[4] for row in rows}:
        intervals = sorted((row[2], row[3]) for row in rows if row[4] == stream)
        gaps_ns += sum(
            max(0, start - previous_end)
            for (_previous_start, previous_end), (start, _end) in itertools.pairwise(
                intervals
            )
        )
    return {
        "count": len(rows),
        "time_ms": sum((end - start) for _, _, start, end, _ in rows) / 1_000_000,
        "launch_gap_ms": gaps_ns / 1_000_000,
        "categories": categories,
    }
