"""Immutable identity required before starting an F2 reproduction run."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RunIdentity:
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


__all__ = ["RunIdentity"]
