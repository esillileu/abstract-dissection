"""Transactional loader for the user-authored F2 research catalog manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .db.repository import CatalogRepository

SECTIONS = (
    "papers",
    "targets",
    "experiment_specs",
    "target_experiments",
    "resources",
    "resource_versions",
    "requirements",
    "execution_plans",
    "plan_experiments",
    "resource_bindings",
    "planned_run_slots",
)

IDENTITY_FIELDS = {
    "papers": ("paper_id",),
    "targets": ("target_id",),
    "experiment_specs": ("experiment_spec_id",),
    "target_experiments": ("target_id", "experiment_spec_id"),
    "resources": ("resource_id",),
    "resource_versions": ("resource_version_id",),
    "requirements": ("requirement_id",),
    "execution_plans": ("execution_plan_id",),
    "plan_experiments": ("plan_experiment_id",),
    "resource_bindings": ("plan_experiment_id", "requirement_id"),
    "planned_run_slots": ("planned_run_slot_id",),
}


def read_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("catalog manifest must be a JSON object")
    unknown = set(payload) - ({"schema_version"} | set(SECTIONS))
    if unknown:
        raise ValueError(f"unknown catalog manifest sections: {sorted(unknown)}")
    for section in SECTIONS:
        rows = payload.get(section, [])
        if not isinstance(rows, list):
            raise ValueError(f"catalog manifest section {section!r} must be a list")
        fields = IDENTITY_FIELDS[section]
        ids = [tuple(row.get(field) for field in fields) for row in rows]
        if any(None in identity for identity in ids) or len(ids) != len(set(ids)):
            raise ValueError(
                f"catalog manifest section {section!r} has missing or duplicate IDs"
            )
    return payload


def digest_manifest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_manifest(repo: CatalogRepository, payload: dict[str, Any]) -> dict[str, int]:
    """Validate references through DB constraints and upsert everything atomically."""
    calls = {
        "papers": repo.upsert_paper,
        "targets": repo.upsert_target,
        "experiment_specs": repo.upsert_experiment_spec,
        "target_experiments": repo.link_target_experiment,
        "resources": repo.upsert_resource,
        "resource_versions": repo.upsert_resource_version,
        "requirements": repo.upsert_requirement,
        "execution_plans": repo.upsert_execution_plan,
        "plan_experiments": repo.upsert_plan_experiment,
        "resource_bindings": repo.bind_plan_requirement,
        "planned_run_slots": repo.upsert_planned_run_slot,
    }
    counts: dict[str, int] = {}
    for section in SECTIONS:
        rows = payload.get(section, [])
        for row in rows:
            calls[section](**row)
        counts[section] = len(rows)
    return counts


__all__ = ["digest_manifest", "load_manifest", "read_manifest"]
