from __future__ import annotations

import csv
import json
from pathlib import Path


def _metric_rows(
    root: Path, *, default_accuracy_split: str | None = None
) -> tuple[list[tuple[int, str, float]], dict[str, float]]:
    path = root / "metrics.csv"
    if not path.is_file():
        return [], {}
    output: list[tuple[int, str, float]] = []
    final: dict[str, float] = {}
    with path.open(encoding="utf-8", newline="") as stream:
        for index, row in enumerate(csv.DictReader(stream)):
            if row.get("metric") and row.get("value") not in {None, ""}:
                pairs = [(str(row["metric"]), row["value"])]
            else:
                pairs = [
                    (key, value)
                    for key, value in row.items()
                    if key
                    not in {
                        "update",
                        "epoch",
                        "plot_index",
                        "condition",
                        "batch_size",
                        "eval_interval",
                    }
                ]
            for key, value in pairs:
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    continue
                split = str(row.get("split", ""))
                if not split and "accuracy" in key.lower():
                    split = default_accuracy_split or ""
                metric = _metric_name(key, split=split)
                if metric.endswith("/accuracy"):
                    # DS1 original e03/e04 record accuracy once per epoch,
                    # while their update counter advances three times per
                    # epoch.  Preserve the graph's epoch axis in the metric
                    # series; zero is a valid first epoch and must not fall
                    # through via truthiness-based selection.
                    step = _int_value(
                        row.get("epoch"),
                        row.get("update"),
                        row.get("plot_index"),
                        default=index,
                    )
                else:
                    step = _int_value(
                        row.get("update"),
                        row.get("epoch"),
                        row.get("plot_index"),
                        default=index,
                    )
                output.append((step, metric, number))
                final[f"final/{metric}"] = number
    return output, final


def _metric_name(key: str, *, split: str = "") -> str:
    lowered = key.lower()
    if "perplexity" in lowered:
        prefix = (
            split.lower() if split.lower() in {"train", "valid", "test"} else "train"
        )
        return f"{prefix}/perplexity"
    if "accuracy" in lowered:
        prefix = (
            split.lower() if split.lower() in {"train", "test", "test-full"} else "test"
        )
        return f"{prefix}/accuracy"
    if "loss" in lowered or "objective" in lowered:
        return "train/loss"
    return f"observation/{key}"


def _int_value(*values, default: int) -> int:
    for value in values:
        if value not in {None, ""}:
            return int(float(value))
    return default


def _last_axis(root: Path, name: str) -> int:
    path = root / "metrics.csv"
    if not path.is_file():
        return 0
    with path.open(encoding="utf-8", newline="") as stream:
        values = [row.get(name) for row in csv.DictReader(stream)]
    return max((_int_value(value, default=0) for value in values), default=0)


def _training_time(root: Path, fallback: float) -> float:
    path = root / "timing.json"
    if not path.is_file():
        return fallback
    return float(json.loads(path.read_text(encoding="utf-8"))["training_wall_time_s"])
