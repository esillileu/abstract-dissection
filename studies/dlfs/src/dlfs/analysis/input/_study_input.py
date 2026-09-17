"""StudyAnalysisInput: Materialized, canonical input for DeepScratch study renderers."""

from __future__ import annotations

import csv
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np

from repro_mlflow.artifact_cache import MlflowArtifactCache

from ...identity import Variant
from ..declarations import StudyDeclaration
from ._model import AnalysisRun
from ._serde import (
    callable_fingerprint,
    decode_history,
    encode_history,
    seed_key,
)
from ._store import PreparedAnalysisStore


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
            else PreparedAnalysisStore(prepared_cache_dir, refresh=refresh_analysis)
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
                key=lambda run: seed_key(run.seed),
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

    def _raw_artifact_file(self, run: AnalysisRun, artifact_path: str) -> Path | None:
        native_path = run.result.artifact_aliases.get(artifact_path, artifact_path)
        materialized = Path(native_path)
        if materialized.is_absolute():
            return materialized if materialized.is_file() else None
        if run.local_artifact_root is not None:
            candidate = run.local_artifact_root / native_path
            if candidate.is_file():
                return candidate
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
                "row_filter": callable_fingerprint(row_filter),
                "x_value": callable_fingerprint(x_value),
                "y_value": callable_fingerprint(y_value),
            },
        )
        cached = None if self._prepared is None else self._prepared.get(prepared_key)
        if isinstance(cached, list):
            return [decode_history(history) for history in cached]
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
                prepared_key, [encode_history(history) for history in histories]
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
            return [decode_history(history) for history in cached]
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
                prepared_key, [encode_history(history) for history in histories]
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
