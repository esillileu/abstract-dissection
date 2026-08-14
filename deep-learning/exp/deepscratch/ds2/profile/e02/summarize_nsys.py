"""Summarize CUDA launch API counts from e02 Nsight SQLite exports."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sqlite3

from exp.deepscratch.ds2.profile.paths import profile_cache


DEFAULT_INPUT = profile_cache("e02") / "nsys"
DEFAULT_OUTPUT = profile_cache("e02") / "nsys/cuda_api_summary.csv"


def summarize_database(path: Path) -> dict[str, int | str]:
    with sqlite3.connect(path) as connection:
        counts = dict(
            connection.execute(
                """
                SELECT strings.value, COUNT(*)
                FROM CUPTI_ACTIVITY_KIND_RUNTIME AS runtime
                JOIN StringIds AS strings ON strings.id = runtime.nameId
                WHERE strings.value = 'cuLaunchKernel'
                   OR strings.value LIKE 'cudaLaunchKernel%'
                GROUP BY strings.value
                """
            )
        )
        total_api_calls = int(
            connection.execute(
                "SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_RUNTIME"
            ).fetchone()[0]
        )
    driver = int(counts.get("cuLaunchKernel", 0))
    runtime = sum(
        int(value)
        for name, value in counts.items()
        if str(name).startswith("cudaLaunchKernel")
    )
    return {
        "condition": path.stem,
        "driver_launch_calls": driver,
        "runtime_launch_calls": runtime,
        "total_launch_calls": driver + runtime,
        "total_cuda_api_calls": total_api_calls,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    rows = [summarize_database(path) for path in sorted(args.input.glob("*.sqlite"))]
    if not rows:
        parser.error(f"no Nsight SQLite exports found under {args.input}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
