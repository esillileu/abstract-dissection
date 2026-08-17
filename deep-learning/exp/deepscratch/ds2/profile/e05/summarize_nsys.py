"""Summarize e05 NVTX operation counts and available CUDA trace data."""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from pathlib import Path

from exp.deepscratch.ds2.profile.paths import (
    profile_analysis,
    profile_artifacts,
    profile_measurements,
)


DEFAULT_INPUT = profile_artifacts("e05") / "nsys"
DEFAULT_OUTPUT = profile_analysis("e05") / "nsys"
DEFAULT_MEASUREMENTS = profile_measurements("e05")


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


def _kernel_summary(
    rows: list[tuple[str, str, int, int, int]]
) -> dict[str, object]:
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
            for (_previous_start, previous_end), (start, _end) in zip(
                intervals, intervals[1:], strict=False
            )
        )
    return {
        "count": len(rows),
        "time_ms": sum((end - start) for _, _, start, end, _ in rows) / 1_000_000,
        "launch_gap_ms": gaps_ns / 1_000_000,
        "categories": categories,
    }


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
        apis = {name: (int(count), float(duration)) for name, count, duration in api_rows}
        launch_count = sum(
            count
            for name, (count, _duration) in apis.items()
            if "LaunchKernel" in name
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
            lstm_kernel_summary["categories"]
            .get("elementwise", {})
            .get("time_ms", 0.0)
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


def run(
    input_dir: Path = DEFAULT_INPUT,
    output_dir: Path = DEFAULT_OUTPUT,
    measurement_dir: Path = DEFAULT_MEASUREMENTS,
) -> tuple[Path, Path]:
    rows = [
        summarize_database(path, measurement_dir)
        for path in sorted(input_dir.glob("*.sqlite"))
    ]
    if not rows:
        raise FileNotFoundError(f"no Nsight SQLite exports under {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "summary.json"
    json_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    csv_path = output_dir / "gemm_counts.csv"
    names = (
        "TimeLSTM/forward_input_gemm",
        "TimeLSTM/backward_dWx_gemm",
        "TimeLSTM/backward_dWh_gemm",
        "TimeLSTM/backward_dX_gemm",
    )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("stage", *names))
        writer.writeheader()
        for row in rows:
            counts = row["timelstm_gemm_nvtx_counts"]
            writer.writerow(
                {"stage": row["stage"], **{name: counts.get(name, 0) for name in names}}
            )
    print(json_path)
    print(csv_path)
    return json_path, csv_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--measurements", type=Path, default=DEFAULT_MEASUREMENTS)
    arguments = parser.parse_args()
    run(arguments.input, arguments.output, arguments.measurements)
