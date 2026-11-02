"""Compatibility wrapper for RunIdentity pointing to canonical execution.identity."""

from __future__ import annotations

from dataclasses import dataclass

from f2.execution.identity import ConditionIdentity


@dataclass(frozen=True)
class RunIdentity:
    """Historical RunIdentity preserving tags and validation."""

    planned_run_slot_id: str
    plan_revision: int
    config_digest: str

    def __post_init__(self) -> None:
        required = {
            "planned_run_slot_id": self.planned_run_slot_id,
            "config_digest": self.config_digest,
        }
        missing = [name for name, value in required.items() if not value.strip()]
        if missing:
            raise ValueError(
                "tracked F2 run identity is incomplete: " + ", ".join(missing)
            )
        if self.plan_revision < 1:
            raise ValueError("plan revision must be positive")

    def tags(self) -> dict[str, str]:
        tags = {
            "f2.planned_run_slot_id": self.planned_run_slot_id,
            "f2.plan_revision": str(self.plan_revision),
            "f2.config_digest": self.config_digest,
        }
        return tags

    def to_condition_identity(
        self,
        suite_id: str = "w2v",
        experiment_id: str = "default",
        condition_id: str = "default",
        seed: int = 1,
    ) -> ConditionIdentity:
        return ConditionIdentity(
            suite_id=suite_id,
            experiment_id=experiment_id,
            condition_id=condition_id,
            seed=seed,
            config_digest=self.config_digest,
            plan_revision=self.plan_revision,
            planned_run_slot_id=self.planned_run_slot_id,
        )


__all__ = ["RunIdentity"]
