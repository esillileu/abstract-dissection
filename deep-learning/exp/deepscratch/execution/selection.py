"""Canonical attempt selection shared by status and analysis consumers."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import inspect
from typing import Any, Iterable

from mlflow.entities import ViewType

from exp.framework.results import NativeRunResult
from exp.framework.results.selection import attempt_priority
from mlprosection_mlflow.artifact_cache import MlflowArtifactCache

from ..identity import Variant, Volume
from ..legacy import LegacyCompatibility
from ..analysis.declarations import MetricDeclaration


@dataclass(frozen=True)
class AttemptRef:
    run_id: str
    namespace: str
    study_id: str
    condition_id: str
    seed: str
    protocol_version: str
    status: str
    start_time: int
    disposition: str | None
    durable_complete: bool | None

    @property
    def is_canonical_namespace(self) -> bool:
        return self.namespace.startswith("deepscratch.")


class CanonicalAttemptSelector:
    """Apply the documented precedence to immutable MLflow attempts."""

    def __init__(self, client, *, tracking_uri: str | None = None) -> None:
        self.client = client
        resolved_tracking_uri = tracking_uri or getattr(client, "tracking_uri", None)
        self._artifact_cache = MlflowArtifactCache(
            client,
            str(resolved_tracking_uri or "http://127.0.0.1:5000"),
        )
        self._legacy = LegacyCompatibility(client, artifact_cache=self._artifact_cache)
        self._attempt_cache: dict[
            tuple[Volume, Variant], tuple[AttemptRef, ...]
        ] = {}

    def attempts(self, volume: Volume, variant: Variant) -> tuple[AttemptRef, ...]:
        coordinate = (volume, variant)
        cached = self._attempt_cache.get(coordinate)
        if cached is not None:
            return cached
        namespace = f"deepscratch.{volume.value}"
        output = []
        experiment = self.client.get_experiment_by_name(namespace)
        if experiment is not None:
            runs = self.client.search_runs(
                [experiment.experiment_id],
                run_view_type=ViewType.ACTIVE_ONLY,
                order_by=["attributes.start_time DESC"],
                max_results=10_000,
            )
            for run in runs:
                tags = run.data.tags
                run_type = tags.get("run.type")
                if run_type not in {"seed_trial", "profile"}:
                    continue
                if tags.get("implementation.variant") != variant.value:
                    continue
                study = _first(tags.get("experiment.id"), tags.get("experiment.ids"), run.data.params.get("experiment_id"), run.data.params.get("run/experiment_id"))
                condition = _first(tags.get("condition.id"), tags.get("atomic_run.id"), run.data.params.get("atomic_run_id"))
                if not study or not condition:
                    continue
                durable = tags.get("result.durable_complete")
                output.append(AttemptRef(
                    run_id=run.info.run_id,
                    namespace=namespace,
                    study_id=study,
                    condition_id=condition,
                    seed=_first(tags.get("seed"), tags.get("master_seed"), run.data.params.get("seed/master"), run.data.params.get("seed")) or "single",
                    protocol_version=tags.get("protocol.version", "legacy"),
                    status=str(run.info.status).upper(),
                    start_time=int(run.info.start_time or 0),
                    disposition=tags.get("transfer.import.disposition"),
                    durable_complete=None if durable is None else durable.lower() == "true",
                ))
        output.extend(
            AttemptRef(**attempt)
            for attempt in self._legacy.attempts(volume, variant)
        )
        attempts = tuple(output)
        self._attempt_cache[coordinate] = attempts
        return attempts

    def load_result(
        self,
        attempt: AttemptRef,
        *,
        volume: Volume,
        variant: Variant,
        declarations: tuple[MetricDeclaration, ...],
    ) -> NativeRunResult:
        """Load a selected result without exposing its storage generation."""
        if not attempt.is_canonical_namespace:
            return self._legacy.load_result(
                attempt.run_id,
                variant=variant,
                declarations=declarations,
            )
        adapter = importlib.import_module(
            f"exp.deepscratch.{volume.value}.{variant.value}.result_adapter"
        )
        kwargs = {}
        if "artifact_cache" in inspect.signature(
            adapter.load_native_result
        ).parameters:
            kwargs["artifact_cache"] = self._artifact_cache
        return adapter.load_native_result(
            self.client, attempt.run_id, declarations, **kwargs
        )

    def select(
        self,
        volume: Volume,
        variant: Variant,
        *,
        study_id: str,
        condition_ids: Iterable[str],
        seed: str | int,
        run_id: str | None = None,
        statuses: tuple[str, ...] = ("FINISHED",),
    ) -> AttemptRef | None:
        aliases = set(condition_ids)
        candidates = [
            item for item in self.attempts(volume, variant)
            if item.study_id == study_id
            and item.condition_id in aliases
            and item.seed == str(seed)
            and item.status in statuses
        ]
        if run_id is not None:
            selected = next((item for item in candidates if item.run_id == run_id), None)
            if selected is None:
                raise ValueError(f"run ID is not in the requested coordinate: {run_id}")
            return selected
        eligible = [item for item in candidates if item.disposition != "imported-alternate"]
        if not eligible:
            return None
        return min(eligible, key=_priority)


def _priority(item: AttemptRef) -> tuple[int, int]:
    return attempt_priority(
        canonical_namespace=item.is_canonical_namespace,
        durable_complete=item.durable_complete,
        disposition=item.disposition,
        start_time=item.start_time,
    )


def _first(*values: Any) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value).split(",")[0]
        if text:
            return text
    return ""
