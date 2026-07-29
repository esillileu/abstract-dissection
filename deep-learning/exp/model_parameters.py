"""Shared model-parameter counting helpers for experiment analysis."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from exp.analyze import RunRef, artifact_file


PARAMETER_MANIFEST_PATH = "model/parameter_manifest.json"


@dataclass(frozen=True)
class ParameterCount:
    """One model size confirmed by one or more seed runs."""

    value: int
    run_count: int


def count_model_parameters(model: Any) -> int:
    """Count unique elements exposed by ``model.named_parameters()``."""

    total = 0
    seen: set[int] = set()
    for _name, parameter in model.named_parameters():
        identity = id(parameter)
        if identity in seen:
            continue
        seen.add(identity)
        total += int(parameter.data.size)
    return total


def count_parameter_manifest(entries: Sequence[Mapping[str, object]]) -> int:
    """Count parameter elements recorded in a schema-v1 manifest."""

    total = 0
    seen_names: set[str] = set()
    for entry in entries:
        try:
            name = entry["name"]
            numel = entry["numel"]
        except (KeyError, TypeError) as exc:
            raise ValueError("invalid model parameter manifest entry") from exc
        if not isinstance(name, str) or not name or name in seen_names:
            raise ValueError(f"duplicate model parameter name: {name!r}")
        if isinstance(numel, bool) or not isinstance(numel, int) or numel < 0:
            raise ValueError(f"invalid parameter count for {name!r}: {numel!r}")
        seen_names.add(name)
        total += numel
    return total


def parameter_count_for_runs(
    client,
    run_refs: Sequence[RunRef],
) -> ParameterCount | None:
    """Load model sizes from seed-run manifests and require consistency."""

    counts = []
    for run in run_refs:
        path = artifact_file(client, run, PARAMETER_MANIFEST_PATH)
        if path is None:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"invalid parameter manifest for run {run.run_id}"
            ) from exc
        if not isinstance(payload, list):
            raise ValueError(f"invalid parameter manifest for run {run.run_id}")
        try:
            counts.append(count_parameter_manifest(payload))
        except ValueError as exc:
            raise ValueError(
                f"invalid parameter manifest for run {run.run_id}"
            ) from exc
    if not counts:
        return None
    unique_counts = sorted(set(counts))
    if len(unique_counts) != 1:
        raise ValueError(
            "parameter counts differ across seed runs: "
            + ", ".join(str(value) for value in unique_counts)
        )
    return ParameterCount(value=unique_counts[0], run_count=len(counts))


def format_parameter_count(count: ParameterCount | None) -> str:
    if count is None:
        return "parameter_count: unavailable"
    return f"parameter_count: {count.value:,} (n={count.run_count} manifests)"


def append_parameter_counts(
    path: Path,
    parameter_counts: Mapping[str, ParameterCount | None],
) -> None:
    """Append model-size rows to either domain's summary CSV schema."""

    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames
    if not fieldnames or "series" not in fieldnames:
        raise ValueError(f"summary CSV has no series column: {path}")
    with path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        for atomic_run_id, parameter_count in parameter_counts.items():
            row = {field: "" for field in fieldnames}
            row["series"] = (
                atomic_run_id
                if "metric" in fieldnames
                else f"{atomic_run_id}/parameter_count"
            )
            if "metric" in fieldnames:
                row["metric"] = "parameter_count"
            if "seed_runs" in fieldnames:
                row["seed_runs"] = (
                    "" if parameter_count is None else parameter_count.run_count
                )
            if "unit" in fieldnames:
                row["unit"] = "parameters"
            if "mean" in fieldnames:
                row["mean"] = (
                    "" if parameter_count is None else parameter_count.value
                )
            writer.writerow(row)
