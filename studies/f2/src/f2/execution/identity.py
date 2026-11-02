"""Condition Identity and Attempt Identity for F2 reproduction suites.

In accordance with ref.md section 4.3:
- Condition Identity: suite, experiment_id, condition_id, seed, semantic config digest,
  and required input/algorithm revision.
- Attempt Identity: Distinguishes physical executions/resumptions of a Condition.
- Tags and MLflow representations derive strictly from canonical identities.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any


def _compute_digest(data: dict[str, Any]) -> str:
    """Compute deterministic SHA-256 hex digest of JSON-serializable dictionary."""
    encoded = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ConditionIdentity:
    """Canonical specification identity for an atomic experimental condition."""

    suite_id: str
    experiment_id: str
    condition_id: str
    seed: int
    config_digest: str
    plan_revision: int = 1
    planned_run_slot_id: str = ""
    extra_dimensions: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.suite_id.strip():
            raise ValueError("suite_id must be non-empty")
        if not self.condition_id.strip():
            raise ValueError("condition_id must be non-empty")
        if not self.config_digest.strip():
            raise ValueError("config_digest must be non-empty")
        if self.plan_revision < 1:
            raise ValueError("plan revision must be positive")
        if not self.planned_run_slot_id:
            object.__setattr__(
                self,
                "planned_run_slot_id",
                f"{self.suite_id}-{self.experiment_id}-{self.condition_id}-s{self.seed}",
            )

    @property
    def identity_key(self) -> str:
        """Deterministic string key representing this condition."""
        return f"{self.planned_run_slot_id}:{self.config_digest[:16]}"

    def tags(self) -> dict[str, str]:
        """MLflow-compatible standard tag mapping for this condition."""
        tags = {
            "f2.suite_id": self.suite_id,
            "f2.experiment_id": self.experiment_id,
            "f2.condition_id": self.condition_id,
            "f2.seed": str(self.seed),
            "f2.planned_run_slot_id": self.planned_run_slot_id,
            "f2.plan_revision": str(self.plan_revision),
            "f2.config_digest": self.config_digest,
        }
        for k, v in self.extra_dimensions.items():
            tags[f"f2.{k}"] = str(v)
        return tags

    @classmethod
    def from_config(
        cls, config: dict[str, Any], suite_id: str | None = None
    ) -> ConditionIdentity:
        """Create ConditionIdentity directly from an executor/run config dictionary."""
        identity = (
            config.get("identity", {})
            if isinstance(config.get("identity"), dict)
            else {}
        )
        slot_id = str(identity.get("planned_run_slot_id", ""))
        config_digest = str(identity.get("config_digest", ""))
        seed = int(identity.get("seed", 1))

        resolved_suite = suite_id or str(identity.get("suite_id", ""))
        if not resolved_suite:
            tracking = config.get("tracking")
            if isinstance(tracking, dict) and "experiment" in tracking:
                exp_str = str(tracking["experiment"])
                resolved_suite = exp_str.rsplit(".", 1)[-1]
        if not resolved_suite:
            resolved_suite = "w2v"

        atomic_run_id = str(config.get("atomic_run_id", ""))
        if "--" in atomic_run_id:
            condition_id = atomic_run_id.split("--", 1)[1]
        elif atomic_run_id:
            condition_id = atomic_run_id
        else:
            condition_id = str(identity.get("experiment_spec_id") or "default")

        plan_id = str(identity.get("execution_plan_id", ""))
        plan_rev = 1
        if "-r" in plan_id:
            try:
                plan_rev = int(plan_id.rsplit("-r", 1)[1])
            except ValueError:
                plan_rev = 1

        experiment_id = str(
            identity.get("study") or identity.get("experiment_id") or "e01"
        )

        return cls(
            suite_id=resolved_suite,
            experiment_id=experiment_id,
            condition_id=condition_id,
            seed=seed,
            config_digest=config_digest,
            plan_revision=plan_rev,
            planned_run_slot_id=slot_id,
        )


@dataclass(frozen=True)
class AttemptIdentity:
    """Identity of a specific physical execution attempt of a condition."""

    condition: ConditionIdentity
    attempt_id: str
    created_at: str

    def __post_init__(self) -> None:
        if not self.attempt_id.strip():
            raise ValueError("attempt_id must be non-empty")

    def run_directory_relpath(self) -> str:
        """Standard relative path under workspace/runs/f2/."""
        return f"{self.condition.suite_id}/{self.condition.planned_run_slot_id}/{self.attempt_id}"


__all__ = [
    "AttemptIdentity",
    "ConditionIdentity",
    "_compute_digest",
]
