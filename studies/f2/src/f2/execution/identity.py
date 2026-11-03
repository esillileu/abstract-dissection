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
        if not self.suite_id or not self.suite_id.strip():
            raise ValueError("suite_id must be non-empty")
        if not self.experiment_id or not self.experiment_id.strip():
            raise ValueError("experiment_id must be non-empty")
        if not self.condition_id or not self.condition_id.strip():
            raise ValueError("condition_id must be non-empty")
        if not self.config_digest or not self.config_digest.strip():
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
        cls,
        config: dict[str, Any],
        suite_id: str | None = None,
        experiment_id: str | None = None,
    ) -> ConditionIdentity:
        """Create ConditionIdentity directly from an executor/run config dictionary."""
        identity = (
            config.get("identity", {})
            if isinstance(config.get("identity"), dict)
            else {}
        )

        resolved_suite = suite_id or str(identity.get("suite_id", "")).strip()
        if not resolved_suite:
            tracking = config.get("tracking")
            if isinstance(tracking, dict) and "experiment" in tracking:
                exp_str = str(tracking["experiment"])
                resolved_suite = exp_str.rsplit(".", 1)[-1].strip()
        if not resolved_suite:
            raise ValueError(
                "ConditionIdentity requires suite_id in config or argument"
            )

        slot_id = str(identity.get("planned_run_slot_id", "")).strip()
        if not slot_id:
            raise ValueError("ConditionIdentity requires identity.planned_run_slot_id")

        config_digest = str(identity.get("config_digest", "")).strip()
        if not config_digest:
            raise ValueError("ConditionIdentity requires identity.config_digest")

        if "seed" not in identity:
            raise ValueError("ConditionIdentity requires identity.seed")
        seed = int(identity["seed"])

        # Ref.md: Extract canonical condition_id directly from normalized identity or atomic_run_id
        condition_id = (
            str(identity.get("condition_id", "")).strip()
            or str(identity.get("experiment_spec_id", "")).strip()
            or str(config.get("atomic_run_id", "")).strip()
        )
        if not condition_id:
            raise ValueError(
                "ConditionIdentity requires identity.condition_id, identity.experiment_spec_id, or atomic_run_id"
            )

        resolved_exp = (
            experiment_id
            or str(identity.get("experiment_id", "")).strip()
            or str(identity.get("study", "")).strip()
            or str(identity.get("execution_plan_id", "")).strip()
        )
        if not resolved_exp:
            raise ValueError(
                "ConditionIdentity requires experiment_id, identity.study, or identity.execution_plan_id"
            )

        raw_rev = identity.get("plan_revision")
        if raw_rev is not None:
            try:
                plan_rev = int(raw_rev)
            except (ValueError, TypeError):
                plan_rev = 1
        else:
            plan_rev = 1

        return cls(
            suite_id=resolved_suite,
            experiment_id=resolved_exp,
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
