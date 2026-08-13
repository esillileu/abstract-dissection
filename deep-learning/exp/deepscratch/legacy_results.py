"""Read-only projection of historical namespaces into DeepScratch identity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient

from .identity import Variant, Volume, legacy_namespace


@dataclass(frozen=True)
class LegacyRunRef:
    run_id: str
    namespace: str
    volume: Volume
    variant: Variant
    experiment_id: str
    condition_id: str
    seed: str
    run_key: str | None
    native_schema: str
    protocol_version: str
    start_time: int
    imported_disposition: str | None


class LegacyResultStore:
    """Centralized legacy namespace and alternate-attempt selection."""

    def __init__(self, client: MlflowClient) -> None:
        self.client = client

    def runs(
        self,
        volume: Volume,
        variant: Variant,
        *,
        experiment_id: str | None = None,
        condition_id: str | None = None,
        seed: str | int | None = None,
    ) -> list[LegacyRunRef]:
        namespace = legacy_namespace(volume, variant)
        experiment = self.client.get_experiment_by_name(namespace)
        if experiment is None:
            return []
        runs = self.client.search_runs(
            [experiment.experiment_id],
            filter_string="attributes.status = 'FINISHED'",
            run_view_type=ViewType.ACTIVE_ONLY,
            order_by=["attributes.start_time DESC"],
            max_results=10_000,
        )
        refs = [self._project(run, namespace, volume, variant) for run in runs]
        return [
            ref for ref in refs
            if (experiment_id is None or ref.experiment_id == experiment_id)
            and (condition_id is None or ref.condition_id == condition_id)
            and (seed is None or ref.seed == str(seed))
        ]

    def select_attempt(
        self,
        volume: Volume,
        variant: Variant,
        *,
        experiment_id: str,
        condition_id: str,
        seed: str | int,
        run_id: str | None = None,
    ) -> LegacyRunRef | None:
        candidates = self.runs(
            volume,
            variant,
            experiment_id=experiment_id,
            condition_id=condition_id,
            seed=seed,
        )
        if run_id is not None:
            selected = next((candidate for candidate in candidates if candidate.run_id == run_id), None)
            if selected is None:
                raise ValueError(f"run ID is not in the requested legacy coordinate: {run_id}")
            return selected
        if not candidates:
            return None
        by_run_key: dict[str, list[LegacyRunRef]] = {}
        for candidate in candidates:
            by_run_key.setdefault(candidate.run_key or candidate.run_id, []).append(candidate)
        canonical_attempts = [self._canonical_same_key(group) for group in by_run_key.values()]
        return max(canonical_attempts, key=lambda candidate: candidate.start_time)

    @staticmethod
    def _canonical_same_key(candidates: list[LegacyRunRef]) -> LegacyRunRef:
        native = [candidate for candidate in candidates if candidate.imported_disposition is None]
        if native:
            return min(native, key=lambda candidate: candidate.start_time)
        primary = [
            candidate for candidate in candidates
            if candidate.imported_disposition == "imported"
        ]
        if primary:
            return min(primary, key=lambda candidate: candidate.start_time)
        # Archives created before disposition tagging may still be unambiguous.
        non_alternate = [
            candidate for candidate in candidates
            if candidate.imported_disposition != "imported-alternate"
        ]
        if non_alternate:
            return min(non_alternate, key=lambda candidate: candidate.start_time)
        raise ValueError(
            "only imported alternate attempts exist; select one by explicit run ID"
        )

    @staticmethod
    def _project(
        run: Any,
        namespace: str,
        volume: Volume,
        variant: Variant,
    ) -> LegacyRunRef:
        tags = run.data.tags
        params = run.data.params
        experiment_id = (
            tags.get("experiment.id")
            or tags.get("experiment.ids", "").split(",")[0]
            or params.get("experiment_id")
            or params.get("run/experiment_id")
            or ""
        )
        condition_id = (
            tags.get("condition.id")
            or tags.get("atomic_run.id")
            or params.get("atomic_run_id")
            or ""
        )
        seed = (
            tags.get("seed")
            or tags.get("master_seed")
            or params.get("seed/master")
            or params.get("seed")
            or "single"
        )
        return LegacyRunRef(
            run_id=run.info.run_id,
            namespace=namespace,
            volume=volume,
            variant=variant,
            experiment_id=str(experiment_id),
            condition_id=str(condition_id),
            seed=str(seed),
            run_key=tags.get("run.key"),
            native_schema=tags.get("result.schema.name", tags.get("schema.version", "legacy")),
            protocol_version=tags.get("protocol.version", "legacy"),
            start_time=int(run.info.start_time or 0),
            imported_disposition=tags.get("transfer.import.disposition"),
        )
