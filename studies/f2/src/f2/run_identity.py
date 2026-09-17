"""Immutable identity required before starting an F2 tracked run attempt."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RunIdentity:
    planned_run_slot_id: str
    plan_revision: int
    config_digest: str
    resource_version_id: str
    resource_manifest_digest: str
    attempt: int = 1
    predecessor_run_id: str | None = None

    def __post_init__(self) -> None:
        required = {
            "planned_run_slot_id": self.planned_run_slot_id,
            "config_digest": self.config_digest,
            "resource_version_id": self.resource_version_id,
            "resource_manifest_digest": self.resource_manifest_digest,
        }
        missing = [name for name, value in required.items() if not value.strip()]
        if missing:
            raise ValueError(
                "tracked F2 run identity is incomplete: " + ", ".join(missing)
            )
        if self.plan_revision < 1 or self.attempt < 1:
            raise ValueError("plan revision and attempt must be positive")
        if self.attempt == 1 and self.predecessor_run_id is not None:
            raise ValueError("the first attempt cannot have a predecessor run")
        if self.attempt > 1 and not self.predecessor_run_id:
            raise ValueError("a resumed attempt requires a predecessor run ID")

    def tags(self) -> dict[str, str]:
        tags = {
            "f2.planned_run_slot_id": self.planned_run_slot_id,
            "f2.plan_revision": str(self.plan_revision),
            "f2.config_digest": self.config_digest,
            "f2.resource_version_id": self.resource_version_id,
            "f2.resource_manifest_digest": self.resource_manifest_digest,
            "f2.attempt": str(self.attempt),
        }
        if self.predecessor_run_id is not None:
            tags["f2.predecessor_run_id"] = self.predecessor_run_id
        return tags


__all__ = ["RunIdentity"]
