"""Generic table analysis utilities and markdown rendering for F2."""

from __future__ import annotations

import csv
import json
import re
import statistics
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")


class BaseEvaluationCache[T]:
    """Signature-checked per-run cache for reconstructible evaluation results."""

    def __init__(
        self, root: Path, signature: dict[str, Any], result_cls: type[T]
    ) -> None:
        self.root = root
        self.signature = signature
        self.result_cls = result_cls

    def _path(self, run_id: str) -> Path:
        if not re.fullmatch(r"[0-9a-zA-Z_-]+", run_id):
            raise ValueError(f"invalid MLflow run ID: {run_id}")
        return self.root / f"eval_{run_id}.json"

    def load(self, run_id: str) -> T | None:
        path = self._path(run_id)
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                return None
            if payload.get("signature") != self.signature:
                return None
            result_data = payload.get("result")
            if not isinstance(result_data, dict):
                return None
            return self.result_cls(**result_data)
        except Exception:
            return None

    def store(self, run_id: str, result: T) -> None:
        path = self._path(run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        res_payload = (
            asdict(result)  # type: ignore[call-overload]
            if is_dataclass(result)
            else result
        )
        payload = {
            "signature": self.signature,
            "result": res_payload,
        }
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def observed_training_seconds(path: Path) -> float:
    """Sum the final observed wall time from each training epoch."""
    target = path / "metrics/observations.csv" if path.is_dir() else path
    if not target.is_file():
        return 0.0
    by_epoch: dict[int, float] = {}
    with target.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            epoch = int(row["epoch"])
            elapsed = float(row["elapsed_seconds"])
            by_epoch[epoch] = max(by_epoch.get(epoch, 0.0), elapsed)
    if not by_epoch:
        return 0.0
    return sum(by_epoch.values())


def format_mean_sd(values: list[float], precision: int = 1) -> str:
    """Format a list of float values as Mean ± SD (or Mean if single value)."""
    if not values:
        return "—"
    m = statistics.fmean(values)
    if len(values) == 1:
        return f"{m:.{precision}f}"
    s = statistics.stdev(values)
    return f"{m:.{precision}f} ± {s:.{precision}f}"


def format_duration(seconds: float) -> str:
    """Format seconds into human-readable duration."""
    hours, remainder = divmod(seconds, 3600.0)
    minutes, remaining_seconds = divmod(remainder, 60.0)
    if hours >= 1.0:
        return f"{int(hours)}h {int(minutes):02d}m {remaining_seconds:04.1f}s"
    if minutes >= 1.0:
        return f"{int(minutes)}m {remaining_seconds:04.1f}s"
    return f"{remaining_seconds:.1f}s"


def format_duration_mean_sd(seconds_list: list[float]) -> str:
    """Format seconds list into Mean ± SD duration string."""
    if not seconds_list:
        return "—"
    m = statistics.fmean(seconds_list)
    if len(seconds_list) == 1:
        return format_duration(m)
    s = statistics.stdev(seconds_list)
    return f"{format_duration(m)} ± {format_duration(s)}"


def write_table_summary(
    path: Path,
    *,
    title: str,
    headers: list[str],
    rows: list[list[str]],
    alignments: list[str] | None = None,
    notes: list[str] | None = None,
    paper_reference: str | None = None,
) -> None:
    """Write a clean, standardized markdown table summary."""
    target_file = path if path.suffix == ".md" else path / "summary.md"
    target_file.parent.mkdir(parents=True, exist_ok=True)
    n_cols = len(headers)
    if alignments is None:
        align_row = ["---:"] * n_cols
        align_row[0] = "---"
    else:
        align_row = [
            "---:" if a == "right" else ("---" if a == "left" else ":---:")
            for a in alignments
        ]

    lines = [
        f"# {title}",
        "",
    ]
    if paper_reference:
        lines.extend([f"**Paper Reference:** {paper_reference}", ""])
    lines.extend(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(align_row) + " |",
        ]
    )
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")

    if notes:
        lines.append("")
        for note in notes:
            lines.append(f"- {note}" if not note.startswith("#") else note)

    target_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_run_records_csv(path: Path, records: list[Any]) -> None:
    """Write run result dataclasses or dictionaries into a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        path.write_text("", encoding="utf-8")
        return
    first = records[0]
    if is_dataclass(first):
        fieldnames = list(first.__dataclass_fields__.keys())
        rows = (asdict(record) for record in records)
    elif isinstance(first, dict):
        fieldnames = list(first.keys())
        rows = records
    else:
        fieldnames = list(getattr(first, "__dict__", {}).keys())
        rows = (getattr(r, "__dict__", {}) for r in records)

    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


__all__ = [
    "BaseEvaluationCache",
    "format_duration",
    "format_duration_mean_sd",
    "format_mean_sd",
    "observed_training_seconds",
    "write_run_records_csv",
    "write_table_summary",
]
