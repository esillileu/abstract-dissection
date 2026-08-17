"""Materialized, canonical input for DeepScratch study renderers.

Renderers consume this model instead of querying MLflow or knowing whether a
selected run came from the canonical or quarantined legacy namespace.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
import marshal
from pathlib import Path
import shutil
from typing import Callable, Mapping, Sequence

import numpy as np

from exp.framework.analysis.core import Curve, aggregate
from exp.framework.paths import WorkspacePaths
from exp.framework.results import NativeRunResult
from mlprosection_mlflow.artifact_cache import MlflowArtifactCache

from ..identity import Variant
from .declarations import StudyDeclaration


@dataclass(frozen=True)
class AnalysisRun:
    run_id: str
    canonical_condition_id: str
    native_condition_id: str
    seed: str
    variant: Variant
    result: NativeRunResult
    local_artifact_root: Path | None = None


class PreparedAnalysisStore:
    """Materialize and replay the renderer-facing analysis inputs."""

    SCHEMA_VERSION = 1

    def __init__(self, root: Path, *, refresh: bool = False) -> None:
        self.root = root
        self.index_path = root / "prepared_analysis.json"
        self._entries: dict[str, object] = {}
        self._dirty = False
        if not refresh:
            try:
                payload = json.loads(self.index_path.read_text(encoding="utf-8"))
                if payload.get("schema_version") == self.SCHEMA_VERSION:
                    self._entries = dict(payload["entries"])
            except (KeyError, OSError, TypeError, json.JSONDecodeError):
                pass

    def key(self, operation: str, payload: object) -> str:
        encoded = json.dumps(
            {"operation": operation, "payload": payload},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return f"{operation}:{hashlib.sha256(encoded).hexdigest()}"

    def get(self, key: str) -> object | None:
        return self._entries.get(key)

    def contains(self, key: str) -> bool:
        return key in self._entries

    def put(self, key: str, value: object) -> None:
        self._entries[key] = value
        self._dirty = True

    def materialize_file(self, key: str, source: Path) -> Path:
        digest = key.rsplit(":", 1)[-1]
        target = self.root / "files" / digest / source.name
        if source.is_dir():
            if target.exists():
                shutil.rmtree(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source, target)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        self.put(key, {"path": str(target.relative_to(self.root))})
        return target

    def cached_file(self, key: str) -> Path | None:
        entry = self.get(key)
        if not isinstance(entry, dict) or "path" not in entry:
            return None
        path = self.root / str(entry["path"])
        return path if path.exists() else None

    def commit(self) -> None:
        if not self._dirty:
            return
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = self.index_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(
                {
                    "schema_version": self.SCHEMA_VERSION,
                    "entries": self._entries,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.index_path)
        self._dirty = False


class StudyAnalysisInput:
    """All selected native results for one study and one variant."""

    def __init__(
        self,
        client,
        declaration: StudyDeclaration,
        variant: Variant,
        runs: Sequence[AnalysisRun],
        *,
        cache_dir: Path,
        tracking_uri: str | None = None,
        refresh_raw: bool = False,
        prepared_cache_dir: Path | None = None,
        refresh_analysis: bool = False,
    ) -> None:
        self._client = client
        self.declaration = declaration
        self.variant = variant
        self._runs = tuple(runs)
        self.cache_dir = cache_dir
        self._artifact_cache = MlflowArtifactCache(
            client, tracking_uri or "default", root=cache_dir
        )
        self._refresh_raw = refresh_raw
        self._refreshed_artifacts: set[tuple[str, str]] = set()
        self._prepared = (
            None
            if prepared_cache_dir is None
            else PreparedAnalysisStore(
                prepared_cache_dir, refresh=refresh_analysis
            )
        )

    def runs(self, condition_ids: Sequence[str]) -> dict[str, list[AnalysisRun]]:
        """Resolve suite-declared aliases to the already selected run set."""
        output: dict[str, list[AnalysisRun]] = {}
        for requested in condition_ids:
            condition = next(
                (
                    item
                    for item in self.declaration.conditions
                    if requested == item.canonical_id
                    or requested in item.implemented_aliases
                    or requested in item.original_aliases
                ),
                None,
            )
            if condition is None:
                output[requested] = []
                continue
            output[requested] = sorted(
                (
                    run
                    for run in self._runs
                    if run.canonical_condition_id == condition.canonical_id
                ),
                key=lambda run: _seed_key(run.seed),
            )
        return output

    def artifact_file(self, run: AnalysisRun, artifact_path: str) -> Path | None:
        prepared_key = self._prepared_key(
            "artifact_file", {"run_id": run.run_id, "path": artifact_path}
        )
        if self._prepared is not None:
            cached = self._prepared.cached_file(prepared_key)
            if cached is not None:
                return cached
            if self._prepared.contains(prepared_key):
                return None
        source = self._raw_artifact_file(run, artifact_path)
        if source is None:
            if self._prepared is not None:
                self._prepared.put(prepared_key, {"missing": True})
            return None
        if self._prepared is None:
            return source
        return self._prepared.materialize_file(prepared_key, source)

    def _raw_artifact_file(
        self, run: AnalysisRun, artifact_path: str
    ) -> Path | None:
        native_path = run.result.artifact_aliases.get(artifact_path, artifact_path)
        materialized = Path(native_path)
        if materialized.is_absolute():
            return materialized if materialized.is_file() else None
        if run.local_artifact_root is not None:
            candidate = run.local_artifact_root / native_path
            if candidate.is_file():
                return candidate
        # Raw downloads are keyed only by the MLflow store and run ID. Changes
        # to analysis declarations or renderer options must not force an
        # unchanged run's artifacts to be fetched again.
        try:
            cache_key = (run.run_id, native_path)
            if self._refresh_raw and cache_key not in self._refreshed_artifacts:
                staged = self._artifact_cache.fetch(run.run_id, native_path)
                try:
                    downloaded = self._artifact_cache.replace(
                        run.run_id, native_path, staged
                    )
                finally:
                    self._artifact_cache.discard(staged)
                self._refreshed_artifacts.add(cache_key)
            else:
                downloaded = self._artifact_cache.get(run.run_id, native_path)
        except Exception:
            return None
        return downloaded if downloaded.is_file() else None

    def artifact_rows(
        self, run: AnalysisRun, artifact_path: str
    ) -> list[dict[str, str]]:
        prepared_key = self._prepared_key(
            "artifact_rows", {"run_id": run.run_id, "path": artifact_path}
        )
        cached = None if self._prepared is None else self._prepared.get(prepared_key)
        if isinstance(cached, list):
            return [dict(row) for row in cached]
        path = self.artifact_file(run, artifact_path)
        if path is None:
            return []
        with path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if self._prepared is not None:
            self._prepared.put(prepared_key, rows)
        return rows

    def histories_from_artifact(
        self,
        runs: Sequence[AnalysisRun],
        *,
        artifact_path: str,
        x: str,
        y: str,
        row_filter: Callable[[Mapping[str, str]], bool] | None = None,
        x_value: Callable[[Mapping[str, str]], float] | None = None,
        y_value: Callable[[Mapping[str, str]], float] | None = None,
    ) -> list[dict[float, float]]:
        prepared_key = self._prepared_key(
            "histories_from_artifact",
            {
                "run_ids": [run.run_id for run in runs],
                "artifact_path": artifact_path,
                "x": x,
                "y": y,
                "row_filter": _callable_fingerprint(row_filter),
                "x_value": _callable_fingerprint(x_value),
                "y_value": _callable_fingerprint(y_value),
            },
        )
        cached = None if self._prepared is None else self._prepared.get(prepared_key)
        if isinstance(cached, list):
            return [_decode_history(history) for history in cached]
        histories = []
        for run in runs:
            history: dict[float, float] = {}
            for row in self.artifact_rows(run, artifact_path):
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
        if self._prepared is not None:
            self._prepared.put(
                prepared_key, [_encode_history(history) for history in histories]
            )
        return histories

    def metric_histories(
        self, runs: Sequence[AnalysisRun], metric_id: str
    ) -> list[dict[float, float]]:
        prepared_key = self._prepared_key(
            "metric_histories",
            {"run_ids": [run.run_id for run in runs], "metric_id": metric_id},
        )
        cached = None if self._prepared is None else self._prepared.get(prepared_key)
        if isinstance(cached, list):
            return [_decode_history(history) for history in cached]
        histories = []
        for run in runs:
            series = run.result.metric(metric_id)
            if series is None:
                values = self._client.get_metric_history(run.run_id, metric_id)
                history = {
                    float(item.step): float(item.value)
                    for item in values
                    if np.isfinite(item.value)
                }
            else:
                history = dict(zip(series.steps, series.values, strict=True))
            if history:
                histories.append(history)
        if self._prepared is not None:
            self._prepared.put(
                prepared_key, [_encode_history(history) for history in histories]
            )
        return histories

    def metric_value(self, run: AnalysisRun, metric_id: str) -> float | None:
        prepared_key = self._prepared_key(
            "metric_value", {"run_id": run.run_id, "metric_id": metric_id}
        )
        cached = None if self._prepared is None else self._prepared.get(prepared_key)
        if isinstance(cached, dict) and "value" in cached:
            value = cached["value"]
            return None if value is None else float(value)
        series = run.result.metric(metric_id)
        if series is not None and series.values:
            value = float(series.values[-1])
        else:
            history = self._client.get_metric_history(run.run_id, metric_id)
            value = None if not history else float(history[-1].value)
        if self._prepared is not None:
            self._prepared.put(prepared_key, {"value": value})
        return value

    def commit_prepared(self) -> None:
        if self._prepared is not None:
            self._prepared.commit()

    def _prepared_key(self, operation: str, payload: object) -> str:
        if self._prepared is None:
            return operation
        return self._prepared.key(operation, payload)


def artifact_file(data: StudyAnalysisInput, run: AnalysisRun, artifact_path: str):
    return data.artifact_file(run, artifact_path)


def artifact_rows(data: StudyAnalysisInput, run: AnalysisRun, artifact_path: str):
    return data.artifact_rows(run, artifact_path)


def histories_from_artifact(data: StudyAnalysisInput, runs, **kwargs):
    return data.histories_from_artifact(runs, **kwargs)


def metric_histories(data: StudyAnalysisInput, runs, metric: str):
    return data.metric_histories(runs, metric)


def curve_from_artifact(data: StudyAnalysisInput, runs, **kwargs) -> Curve:
    return aggregate(data.histories_from_artifact(runs, **kwargs))


def local_artifact_root(client, run_id: str) -> Path | None:
    """Resolve only the canonical staging location advertised by run tags."""
    run = client.get_run(run_id)
    tags = run.data.tags
    run_key = tags.get("run.key")
    required = (
        tags.get("domain.name"),
        tags.get("suite.name"),
        tags.get("experiment.id"),
        tags.get("implementation.variant"),
        run_key,
    )
    if not all(required):
        return None
    path = WorkspacePaths.from_environment(Path.cwd()).run_staging(
        domain=str(required[0]),
        suite=str(required[1]),
        study=str(required[2]),
        variant=str(required[3]),
        run_key=str(required[4]),
    ) / "record"
    return path if path.is_dir() else None


def _seed_key(value: str) -> tuple[int, str]:
    return (0, f"{int(value):020d}") if value.isdigit() else (1, value)


def _encode_history(history: Mapping[float, float]) -> list[list[float]]:
    return [[float(step), float(value)] for step, value in history.items()]


def _decode_history(payload: object) -> dict[float, float]:
    if not isinstance(payload, list):
        return {}
    return {float(item[0]): float(item[1]) for item in payload}


def _callable_fingerprint(function: Callable | None) -> str | None:
    if function is None:
        return None
    code = getattr(function, "__code__", None)
    if code is None:
        return repr(function)
    closure = tuple(
        repr(cell.cell_contents) for cell in (getattr(function, "__closure__", None) or ())
    )
    payload = marshal.dumps(code) + repr(closure).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "AnalysisRun",
    "StudyAnalysisInput",
    "artifact_file",
    "artifact_rows",
    "curve_from_artifact",
    "histories_from_artifact",
    "local_artifact_root",
    "metric_histories",
]
