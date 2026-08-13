"""Compare declared DeepScratch plans with MLflow run state."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Literal, Sequence

from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient

from exp.domain import RunPlan

from .identity import Variant, Volume, legacy_namespace


PlanStatus = Literal["completed", "running", "failed", "missing"]


@dataclass(frozen=True)
class PlannedRunStatus:
    experiment_id: str
    condition_id: str
    seed: int | None
    variant: str
    status: PlanStatus
    run_id: str | None = None
    mlflow_status: str | None = None
    namespace: str | None = None
    attempt_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PlanStatusReport:
    volume: str
    variant: str
    entries: tuple[PlannedRunStatus, ...]

    @property
    def counts(self) -> dict[str, int]:
        raw = Counter(entry.status for entry in self.entries)
        return {
            status: raw.get(status, 0)
            for status in ("completed", "running", "failed", "missing")
        }

    @property
    def incomplete_count(self) -> int:
        return len(self.entries) - self.counts["completed"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "volume": self.volume,
            "variant": self.variant,
            "planned": len(self.entries),
            "incomplete": self.incomplete_count,
            "counts": self.counts,
            "entries": [entry.to_dict() for entry in self.entries],
        }


@dataclass(frozen=True)
class _Attempt:
    run_id: str
    namespace: str
    experiment_id: str
    condition_id: str
    seed: str
    protocol_version: str
    status: str
    start_time: int


def inspect_plan_status(
    client: MlflowClient,
    plans: Sequence[RunPlan],
    *,
    volume: Volume,
    variant: Variant,
    expected_protocols: dict[tuple[str, str], str] | None = None,
) -> PlanStatusReport:
    """Classify each plan entry across the new and historical namespaces."""
    attempts = _attempts(client, volume=volume, variant=variant)
    indexed: dict[tuple[str, str, str], list[_Attempt]] = {}
    for attempt in attempts:
        expected_protocol = (expected_protocols or {}).get(
            (attempt.experiment_id, attempt.condition_id)
        )
        if (
            expected_protocol is not None
            and attempt.protocol_version != expected_protocol
            and not (
                variant is Variant.ORIGINAL
                and attempt.protocol_version == "legacy"
                and not attempt.namespace.startswith("deepscratch.")
                and expected_protocol == "book-source-v1"
            )
        ):
            continue
        key = (attempt.experiment_id, attempt.condition_id, attempt.seed)
        indexed.setdefault(key, []).append(attempt)

    entries = []
    for plan in plans:
        seed = "single" if plan.seed is None else str(plan.seed)
        matches = indexed.get(
            (plan.experiment_id, plan.atomic_run_id, seed), []
        )
        entries.append(
            _classify(plan, matches, variant=variant, seed=seed)
        )
    return PlanStatusReport(volume.value, variant.value, tuple(entries))


def _attempts(
    client: MlflowClient,
    *,
    volume: Volume,
    variant: Variant,
) -> list[_Attempt]:
    namespaces = (
        f"deepscratch.{volume.value}",
        legacy_namespace(volume, variant),
    )
    attempts: list[_Attempt] = []
    for namespace in namespaces:
        try:
            experiment = client.get_experiment_by_name(namespace)
        except Exception as exc:
            raise RuntimeError(
                f"failed to inspect MLflow experiment {namespace}: {exc}"
            ) from exc
        if experiment is None:
            continue
        try:
            runs = client.search_runs(
                [experiment.experiment_id],
                run_view_type=ViewType.ACTIVE_ONLY,
                order_by=["attributes.start_time DESC"],
                max_results=10_000,
            )
        except Exception as exc:
            raise RuntimeError(
                f"failed to inspect MLflow runs in {namespace}: {exc}"
            ) from exc
        for run in runs:
            tags = run.data.tags
            if tags.get("run.type") != "seed_trial":
                continue
            if (
                namespace.startswith("deepscratch.")
                and tags.get("implementation.variant") != variant.value
            ):
                continue
            if tags.get("transfer.import.disposition") == "imported-alternate":
                continue
            experiment_id = _experiment_id(run)
            condition_id = _condition_id(run)
            if not experiment_id or not condition_id:
                continue
            attempts.append(_Attempt(
                run_id=run.info.run_id,
                namespace=namespace,
                experiment_id=experiment_id,
                condition_id=condition_id,
                seed=_seed(run),
                protocol_version=tags.get("protocol.version", "legacy"),
                status=str(run.info.status).upper(),
                start_time=int(run.info.start_time or 0),
            ))
    return attempts


def _classify(
    plan: RunPlan,
    attempts: Sequence[_Attempt],
    *,
    variant: Variant,
    seed: str,
) -> PlannedRunStatus:
    priority = ("FINISHED", "RUNNING", "SCHEDULED", "FAILED", "KILLED")
    selected = None
    for status in priority:
        candidates = [attempt for attempt in attempts if attempt.status == status]
        if candidates:
            selected = max(candidates, key=lambda attempt: attempt.start_time)
            break
    if selected is None and attempts:
        selected = max(attempts, key=lambda attempt: attempt.start_time)

    if selected is None:
        status: PlanStatus = "missing"
    elif selected.status == "FINISHED":
        status = "completed"
    elif selected.status in {"RUNNING", "SCHEDULED"}:
        status = "running"
    else:
        status = "failed"
    return PlannedRunStatus(
        experiment_id=plan.experiment_id,
        condition_id=plan.atomic_run_id,
        seed=None if seed == "single" else int(seed),
        variant=variant.value,
        status=status,
        run_id=None if selected is None else selected.run_id,
        mlflow_status=None if selected is None else selected.status,
        namespace=None if selected is None else selected.namespace,
        attempt_count=len(attempts),
    )


def _experiment_id(run: Any) -> str:
    tags = run.data.tags
    params = run.data.params
    return str(
        tags.get("experiment.id")
        or tags.get("experiment.ids", "").split(",")[0]
        or params.get("experiment_id")
        or params.get("run/experiment_id")
        or ""
    )


def _condition_id(run: Any) -> str:
    tags = run.data.tags
    params = run.data.params
    return str(
        tags.get("condition.id")
        or tags.get("atomic_run.id")
        or params.get("atomic_run_id")
        or ""
    )


def _seed(run: Any) -> str:
    tags = run.data.tags
    params = run.data.params
    value = (
        tags.get("seed")
        or tags.get("master_seed")
        or params.get("seed/master")
        or params.get("seed")
    )
    return "single" if value is None else str(value)
