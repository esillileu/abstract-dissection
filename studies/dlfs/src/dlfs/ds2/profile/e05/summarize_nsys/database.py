"""Nsight Systems SQLite database trace summarization."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from dlfs.ds2.profile.paths import profile_measurements

from .kernels import (
    _has_table,
    _kernel_summary,
    _kernels_in_nested_ranges,
    _kernels_in_ranges,
)

DEFAULT_MEASUREMENTS = profile_measurements("e05")


def summarize_database(
    path: Path,
    measurement_dir: Path = DEFAULT_MEASUREMENTS,
) -> dict[str, object]:
    with sqlite3.connect(path) as connection:
        nvtx = dict(
            connection.execute(
                """
                SELECT text, COUNT(*)
                FROM NVTX_EVENTS
                WHERE text LIKE 'TimeLSTM/%gemm'
                GROUP BY text
                """
            )
        )
        phase_rows = connection.execute(
            """
            SELECT text, COUNT(*), AVG(end - start) / 1000000.0
            FROM NVTX_EVENTS
            WHERE text LIKE 'e05/full_update/%' AND end IS NOT NULL
            GROUP BY text
            """
        ).fetchall()
        api_rows = connection.execute(
            """
            SELECT strings.value, COUNT(*), SUM(runtime.end - runtime.start) / 1000000.0
            FROM CUPTI_ACTIVITY_KIND_RUNTIME AS runtime
            JOIN StringIds AS strings ON strings.id = runtime.nameId
            GROUP BY strings.value
            """
        ).fetchall()
        apis = {
            name: (int(count), float(duration)) for name, count, duration in api_rows
        }
        launch_count = sum(
            count for name, (count, _duration) in apis.items() if "LaunchKernel" in name
        )
        allocation_count = sum(
            count
            for name, (count, _duration) in apis.items()
            if "Malloc" in name or "Free" in name or "MemAlloc" in name
        )
        diagnostics = [
            text
            for (text,) in connection.execute(
                "SELECT text FROM DIAGNOSTIC_EVENT WHERE severity >= 2"
            )
            if "CUDA" in text or "driver" in text
        ]
        kernel_table = next(
            (
                name
                for name in (
                    "CUPTI_ACTIVITY_KIND_KERNEL",
                    "CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL",
                )
                if _has_table(connection, name)
            ),
            None,
        )
        phase_kernels = []
        phase_kernels_by_name = {}
        lstm_kernels = []
        if kernel_table is not None:
            phase_kernels = _kernels_in_ranges(connection, "e05/full_update/%")
            phase_kernels_by_name = {
                name: _kernel_summary(_kernels_in_ranges(connection, name))
                for name, _count, _mean_ms in phase_rows
            }
            lstm_kernels = _kernels_in_nested_ranges(
                connection,
                "TimeLSTM/%_recurrent_loop",
                "e05/full_update/%",
            )
        traced_updates = int(
            connection.execute(
                """
                SELECT COUNT(*) / 2
                FROM NVTX_EVENTS AS inner_ranges
                WHERE inner_ranges.text = 'TimeLSTM/forward_recurrent_loop'
                  AND EXISTS (
                    SELECT 1 FROM NVTX_EVENTS AS outer_ranges
                    WHERE outer_ranges.globalTid = inner_ranges.globalTid
                      AND inner_ranges.start BETWEEN outer_ranges.start AND outer_ranges.end
                      AND outer_ranges.text = 'e05/full_update/model_forward'
                  )
                """
            ).fetchone()[0]
        )
        phase_kernel_summary = _kernel_summary(phase_kernels)
        lstm_kernel_summary = _kernel_summary(lstm_kernels)
        allocation_in_phases = int(
            connection.execute(
                """
                SELECT COUNT(*)
                FROM CUPTI_ACTIVITY_KIND_RUNTIME AS runtime
                JOIN StringIds AS names ON names.id = runtime.nameId
                JOIN NVTX_EVENTS AS ranges
                  ON ranges.globalTid = runtime.globalTid
                 AND runtime.start BETWEEN ranges.start AND ranges.end
                WHERE ranges.text LIKE 'e05/full_update/%'
                  AND (names.value LIKE '%Malloc%'
                    OR names.value LIKE '%Free%'
                    OR names.value LIKE '%MemAlloc%')
                """
            ).fetchone()[0]
        )
        elementwise_ms = float(
            lstm_kernel_summary["categories"].get("elementwise", {}).get("time_ms", 0.0)
        )
        benchmark_path = measurement_dir / path.stem / "benchmark.json"
        steady_update_ms = None
        if benchmark_path.exists():
            benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
            steady_update_ms = float(benchmark["full_update"]["window"]["mean_ms"])
    return {
        "stage": path.stem,
        "timelstm_gemm_nvtx_counts": {name: int(count) for name, count in nvtx.items()},
        "full_update_nvtx": {
            name: {"count": int(count), "mean_host_range_ms": float(mean_ms)}
            for name, count, mean_ms in phase_rows
        },
        "cuda_launch_api_calls": launch_count,
        "allocation_api_calls": allocation_count,
        "kernel_activity_available": kernel_table is not None,
        "kernel_table": kernel_table,
        "full_update_kernels": phase_kernel_summary,
        "full_update_kernels_by_phase": phase_kernels_by_name,
        "lstm_recurrent_kernels": lstm_kernel_summary,
        "traced_full_updates": traced_updates,
        "steady_full_update_ms": steady_update_ms,
        "lstm_elementwise_ms_per_update": (
            elementwise_ms / traced_updates if traced_updates else None
        ),
        "lstm_elementwise_fraction_of_steady_full_update": (
            elementwise_ms / traced_updates / steady_update_ms
            if traced_updates and steady_update_ms
            else None
        ),
        "allocation_api_calls_in_full_update_phases": allocation_in_phases,
        "trace_limitations": diagnostics,
    }
