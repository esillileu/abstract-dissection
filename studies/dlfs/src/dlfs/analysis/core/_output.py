"""Summary CSV writing and experiment selection parsing."""

from __future__ import annotations

import csv
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from repro_core.analysis.core import Curve


def write_summary(path: Path, curves: Mapping[str, Curve]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "series",
                "seed_runs",
                "points",
                "final_mean",
                "final_min",
                "final_max",
            ],
        )
        writer.writeheader()
        for name, curve in curves.items():
            writer.writerow(
                {
                    "series": name,
                    "seed_runs": curve.run_count,
                    "points": len(curve.steps),
                    "final_mean": curve.mean[-1] if len(curve.mean) else "",
                    "final_min": curve.minimum[-1] if len(curve.minimum) else "",
                    "final_max": curve.maximum[-1] if len(curve.maximum) else "",
                }
            )
    return path


def parse_experiment_selection(
    values: Sequence[str], available: Iterable[str]
) -> tuple[list[str], list[str]]:
    """Expand 01, e01, 01-08, and comma-separated analysis selections."""
    supported = tuple(available)
    if not values or any(value.lower() == "all" for value in values):
        return list(supported), []
    requested: list[str] = []
    for value in values:
        for item in value.split(","):
            item = item.strip().lower()
            match = re.fullmatch(r"e?(\d+)(?:-e?(\d+))?", item)
            if match is None:
                raise ValueError(f"invalid experiment selection: {item}")
            start = int(match.group(1))
            end = int(match.group(2)) if match.group(2) is not None else start
            if start > end:
                raise ValueError(f"experiment range must be ascending: {item}")
            requested.extend(f"e{number:02d}" for number in range(start, end + 1))
    unique = list(dict.fromkeys(requested))
    return [item for item in unique if item in supported], [
        item for item in unique if item not in supported
    ]
