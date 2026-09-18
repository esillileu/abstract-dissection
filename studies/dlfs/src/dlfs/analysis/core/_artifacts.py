"""Artifact retrieval helpers: file resolution, CSV loading, and metric histories."""

from __future__ import annotations

import csv
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np

from ._client import RunRef


def artifact_file(client, run: RunRef, artifact_path: str) -> Path | None:
    """Resolve the local schema-v1 mirror, then fall back to client download."""
    if run.local_artifact_root is not None:
        local_path = run.local_artifact_root / artifact_path
        if local_path.is_file():
            return local_path
    if client is None:
        return None
    try:
        if hasattr(client, "artifact_file") and callable(client.artifact_file):
            result = client.artifact_file(run, artifact_path)
            if result is not None and Path(result).is_file():
                return Path(result)
        if hasattr(client, "get") and callable(client.get):
            result = client.get(run.run_id, artifact_path)
            if result is not None and Path(result).is_file():
                return Path(result)
        if hasattr(client, "download_artifacts") and callable(
            client.download_artifacts
        ):
            temp_dir = tempfile.mkdtemp(prefix="repro_artifact_")
            result = client.download_artifacts(run.run_id, artifact_path, temp_dir)
            if result is not None and Path(result).is_file():
                return Path(result)
    except Exception:
        return None
    return None


def artifact_rows(client, run: RunRef, artifact_path: str) -> list[dict[str, str]]:
    """Load one CSV artifact. A missing artifact represents no history."""
    local_path = artifact_file(client, run, artifact_path)
    if local_path is None:
        return []
    with local_path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def histories_from_artifact(
    client,
    runs: Sequence[RunRef],
    *,
    artifact_path: str,
    x: str,
    y: str,
    row_filter: Callable[[Mapping[str, str]], bool] | None = None,
    x_value: Callable[[Mapping[str, str]], float] | None = None,
    y_value: Callable[[Mapping[str, str]], float] | None = None,
) -> list[dict[float, float]]:
    histories: list[dict[float, float]] = []
    for run in runs:
        history: dict[float, float] = {}
        for row in artifact_rows(client, run, artifact_path):
            if row_filter is not None and not row_filter(row):
                continue
            try:
                step = x_value(row) if x_value is not None else float(row[x])
                value = y_value(row) if y_value is not None else float(row[y])
            except (KeyError, TypeError, ValueError):
                continue
            if np.isfinite(step) and np.isfinite(value):
                history[float(step)] = float(value)
        if history:
            histories.append(history)
    return histories


def metric_histories(
    client, runs: Sequence[RunRef], metric: str
) -> list[dict[float, float]]:
    histories = []
    for run in runs:
        values = {
            float(item.step): float(item.value)
            for item in client.get_metric_history(run.run_id, metric)
            if np.isfinite(item.value)
        }
        if values:
            histories.append(values)
    return histories
