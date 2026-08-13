"""Shared helpers for summaries of fixed-seed original-code results."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path

from exp.deepscratch.original_runtime.cache_protocol import cache_is_valid, load_csv


SUMMARY_FIELDS = (
    "series",
    "metric",
    "seed_runs",
    "unit",
    "mean",
    "standard_deviation",
    "minimum",
    "maximum",
)


@dataclass(frozen=True)
class OriginalMetric:
    name: str
    value: float
    unit: str
    decimals: int
    scale: float = 1.0


def last_value(
    rows: list[dict[str, str]],
    key: str,
    **filters: str,
) -> float | None:
    values = []
    for position, row in enumerate(rows):
        if any(row.get(name) != value for name, value in filters.items()):
            continue
        try:
            step = float(
                row.get("update")
                or row.get("epoch")
                or row.get("plot_index")
                or position
            )
            value = float(row[key])
        except (KeyError, TypeError, ValueError):
            continue
        values.append((step, position, value))
    return max(values)[2] if values else None


def summarize_trial(
    directory: Path,
    extractor,
) -> tuple[dict[str, object] | None, list[OriginalMetric]]:
    if not cache_is_valid(directory):
        return None, []
    try:
        manifest = json.loads(
            (directory / "manifest.json").read_text(encoding="utf-8")
        )
        rows = load_csv(directory / "metrics.csv")
    except (OSError, json.JSONDecodeError):
        return None, []
    return manifest, extractor(rows)


def write_experiment_summary(
    *,
    experiment: str,
    root: Path,
    trial_ids: tuple[str, ...],
    extractor,
) -> Path:
    output = root / "image" / f"{experiment}_summary.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    csv_rows = []
    print(f"{experiment} original summary (fixed seed=1)")
    for trial_id in trial_ids:
        directory = root / "data" / experiment / trial_id
        manifest, metrics = summarize_trial(directory, extractor)
        backend = "unknown" if manifest is None else str(manifest.get("backend", "unknown"))
        print(f"[{trial_id}] backend={backend}")
        if not metrics:
            print("completed metrics: unavailable")
        for metric in metrics:
            displayed = metric.value * metric.scale
            text = f"{displayed:.{metric.decimals}f}"
            suffix = " (%)" if metric.unit == "percent" else ""
            print(f"{metric.name}{suffix}: {text}")
            csv_rows.append(
                {
                    "series": trial_id,
                    "metric": metric.name,
                    "seed_runs": 1,
                    "unit": metric.unit,
                    "mean": text,
                    "standard_deviation": f"{0.0:.{metric.decimals}f}",
                    "minimum": text,
                    "maximum": text,
                }
            )
        timing = _json_object(directory / "timing.json")
        parameter_manifest = _json_object(
            directory / "parameter_manifest.json"
        )
        training_time = _number(timing, "training_wall_time_s")
        parameter_count = _integer(parameter_manifest, "parameter_count")
        csv_rows.append(
            _optional_scalar_row(
                series=trial_id,
                metric="training_time_s",
                unit="seconds",
                value=training_time,
                decimals=1,
            )
        )
        csv_rows.append(
            _optional_scalar_row(
                series=trial_id,
                metric="parameter_count",
                unit="parameters",
                value=parameter_count,
                decimals=0,
                standard_deviation=False,
            )
        )
        print(
            "training_time_s: unavailable"
            if training_time is None
            else f"training_time_s: {training_time:.1f}"
        )
        print(
            "parameter_count: unavailable"
            if parameter_count is None
            else f"parameter_count: {parameter_count}"
        )
    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(output)
    return output


def _json_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _number(value: dict[str, object], key: str) -> float | None:
    try:
        return float(value[key])
    except (KeyError, TypeError, ValueError):
        return None


def _integer(value: dict[str, object], key: str) -> int | None:
    try:
        return int(value[key])
    except (KeyError, TypeError, ValueError):
        return None


def _optional_scalar_row(
    *,
    series: str,
    metric: str,
    unit: str,
    value: float | int | None,
    decimals: int,
    standard_deviation: bool = True,
) -> dict[str, object]:
    if value is None:
        return {
            "series": series,
            "metric": metric,
            "seed_runs": "",
            "unit": unit,
            "mean": "",
            "standard_deviation": "",
            "minimum": "",
            "maximum": "",
        }
    text = f"{value:.{decimals}f}"
    return {
        "series": series,
        "metric": metric,
        "seed_runs": 1,
        "unit": unit,
        "mean": text,
        "standard_deviation": (
            f"{0.0:.{decimals}f}" if standard_deviation else ""
        ),
        "minimum": text,
        "maximum": text,
    }
