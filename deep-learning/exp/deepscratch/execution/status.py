"""Compare declared DeepScratch plans with MLflow run state."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import importlib
from typing import Any, Literal, Sequence

from mlflow.tracking import MlflowClient
from exp.framework.execution import RunPlan

from ..identity import Variant, Volume
from .selection import CanonicalAttemptSelector


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
    """Classify each plan entry in the canonical namespace."""
    attempts = _attempts(client, volume=volume, variant=variant)
    indexed: dict[tuple[str, str, str], list[_Attempt]] = {}
    for attempt in attempts:
        expected_protocol = (expected_protocols or {}).get(
            (attempt.experiment_id, attempt.condition_id)
        )
        if (
            expected_protocol is not None
            and attempt.protocol_version != expected_protocol
            and not _protocol_compatible(
                volume,
                attempt.experiment_id,
                attempt.protocol_version,
                expected_protocol,
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


def _protocol_compatible(
    volume: Volume,
    study_id: str,
    actual: str,
    expected: str,
) -> bool:
    schema = importlib.import_module(
        f"exp.deepscratch.{volume.value}.result_schema"
    )
    pairs = schema.PROTOCOL_EQUIVALENCE.get(study_id, ())
    return any({actual, expected} <= set(pair) for pair in pairs)


def _attempts(
    client: MlflowClient,
    *,
    volume: Volume,
    variant: Variant,
) -> list[_Attempt]:
    try:
        selected = CanonicalAttemptSelector(client).attempts(volume, variant)
    except Exception as exc:
        raise RuntimeError(f"failed to inspect MLflow attempts: {exc}") from exc
    output = [
        _Attempt(
            run_id=item.run_id,
            namespace=item.namespace,
            experiment_id=item.study_id,
            condition_id=item.condition_id,
            seed=item.seed,
            protocol_version=item.protocol_version,
            status=item.status,
            start_time=item.start_time,
        )
        for item in selected
        if item.disposition != "imported-alternate"
    ]
    return output


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
