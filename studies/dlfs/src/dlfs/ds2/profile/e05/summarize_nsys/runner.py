"""Execution runner and CSV/JSON reporting for e05 Nsight trace summaries."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from dlfs.ds2.profile.paths import (
    profile_analysis,
    profile_artifacts,
    profile_measurements,
)

from .database import summarize_database

DEFAULT_INPUT = profile_artifacts("e05") / "nsys"
DEFAULT_OUTPUT = profile_analysis("e05") / "nsys"
DEFAULT_MEASUREMENTS = profile_measurements("e05")


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
    json_path.write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
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
