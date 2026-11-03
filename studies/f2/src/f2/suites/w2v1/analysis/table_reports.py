"""Report generation and formatting utilities for W2V1 tables."""

from __future__ import annotations

import csv
import statistics
from pathlib import Path
from typing import Any


def fmt_stats(values: list[float], precision: int = 1) -> str:
    if not values:
        return "—"
    m = statistics.mean(values)
    if len(values) == 1:
        return f"{m:.{precision}f}"
    s = statistics.stdev(values)
    return f"{m:.{precision}f} ± {s:.{precision}f}"


def write_markdown_and_csv(
    md_path: Path,
    csv_path: Path,
    title: str,
    headers: list[str],
    rows: list[list[str]],
    raw_dicts: list[dict[str, Any]],
) -> None:
    md_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        f"# {title}",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if raw_dicts:
        fieldnames = list(raw_dicts[0].keys())
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(raw_dicts)


__all__ = ["fmt_stats", "write_markdown_and_csv"]
