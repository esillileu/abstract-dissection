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
    "requirement_candidates",
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
    "requirement_candidates": ("requirement_id", "resource_id"),
    "execution_plans": ("execution_plan_id",),
    "plan_experiments": ("plan_experiment_id",),
    "resource_bindings": ("plan_experiment_id", "requirement_id"),
    "planned_run_slots": ("planned_run_slot_id",),
}


def _index(rows: list[dict[str, Any]], field: str) -> dict[str, dict[str, Any]]:
    return {row[field]: row for row in rows}


def _expand_planned_run_matrices(payload: dict[str, Any]) -> None:
    """Expand compact, explicit condition/seed matrices into immutable slots."""
    matrices = payload.pop("planned_run_matrices", [])
    if not isinstance(matrices, list):
        raise ValueError(
            "catalog manifest section 'planned_run_matrices' must be a list"
        )
    slots = payload.setdefault("planned_run_slots", [])
    if not isinstance(slots, list):
        raise ValueError("catalog manifest section 'planned_run_slots' must be a list")
    for matrix in matrices:
        plan_experiment_id = str(matrix["plan_experiment_id"])
        execution_plan_id = str(matrix["execution_plan_id"])
        seeds = matrix.get("seeds")
        conditions = matrix.get("conditions")
        if conditions is None and matrix.get("dimensions") is not None:
            conditions = [
                {
                    "atomic_run_id": f"d{int(dimension)}-w{int(tokens) // 1_000_000}m",
                    "training_tokens": int(tokens),
                    "embedding_dimension": int(dimension),
                    "epochs": int(matrix["epochs"]),
                }
                for dimension in matrix["dimensions"]
                for tokens in matrix["training_token_budgets"]
            ]
        if conditions is None and matrix.get("atomic_run_ids") is not None:
            conditions = [
                {
                    "atomic_run_id": str(atomic_run_id),
                    "training_tokens": int(matrix["training_tokens"]),
                    "embedding_dimension": int(matrix["embedding_dimension"]),
                    "epochs": int(matrix["epochs"]),
                }
                for atomic_run_id in matrix["atomic_run_ids"]
            ]
        if not isinstance(seeds, list) or not seeds:
            raise ValueError("planned run matrix requires non-empty seeds")
        if not isinstance(conditions, list) or not conditions:
            raise ValueError("planned run matrix requires non-empty conditions")
        for condition in conditions:
            atomic_run_id = str(condition["atomic_run_id"])
            for seed_value in seeds:
                seed = int(seed_value)
                parameters = {
                    "classification": str(matrix["classification"]),
                    "training_tokens": int(condition["training_tokens"]),
                    "embedding_dimension": int(condition["embedding_dimension"]),
                    "epochs": int(condition["epochs"]),
                    "estimated_token_updates": int(condition["training_tokens"])
                    * int(condition["epochs"]),
                    "device": str(matrix.get("device", "cpu")),
                    "threads": int(matrix.get("threads", 1)),
                    "requires_approval": True,
                }
                slots.append(
                    {
                        "planned_run_slot_id": (
                            f"{execution_plan_id}-{atomic_run_id}-s{seed}"
                        ),
                        "plan_experiment_id": plan_experiment_id,
                        "slot_key": f"{atomic_run_id}-s{seed}",
                        "atomic_run_id": atomic_run_id,
                        "variant_key": atomic_run_id,
                        "seed": seed,
                        "parameters": parameters,
                        "expected": True,
                        "reference_mlflow_run_id": None,
                        "notes": str(matrix["notes"]),
                    }
                )


def _validate_references(payload: dict[str, Any]) -> None:
    papers = _index(payload.get("papers", []), "paper_id")
    targets = _index(payload.get("targets", []), "target_id")
    experiments = _index(payload.get("experiment_specs", []), "experiment_spec_id")
    resources = _index(payload.get("resources", []), "resource_id")
    versions = _index(payload.get("resource_versions", []), "resource_version_id")
    requirements = _index(payload.get("requirements", []), "requirement_id")
    plan_experiments = _index(payload.get("plan_experiments", []), "plan_experiment_id")
    plans = _index(payload.get("execution_plans", []), "execution_plan_id")

    def require(section: str, row: dict[str, Any], field: str, index: dict[str, Any]):
        value = row.get(field)
        if value is not None and value not in index:
            raise ValueError(f"{section} references unknown {field} {value!r}")

    for row in payload.get("targets", []):
        require("targets", row, "paper_id", papers)
    for row in payload.get("experiment_specs", []):
        require("experiment_specs", row, "paper_id", papers)
    for row in payload.get("target_experiments", []):
        require("target_experiments", row, "target_id", targets)
        require("target_experiments", row, "experiment_spec_id", experiments)
    for row in payload.get("resource_versions", []):
        require("resource_versions", row, "resource_id", resources)
    for row in payload.get("resources", []):
        require("resources", row, "canonical_version_id", versions)
        version_id = row.get("canonical_version_id")
        if version_id and versions[version_id]["resource_id"] != row["resource_id"]:
            raise ValueError(
                f"canonical version {version_id!r} does not belong to resource "
                f"{row['resource_id']!r}"
            )
    for row in payload.get("requirements", []):
        require("requirements", row, "experiment_spec_id", experiments)
        require("requirements", row, "required_resource_id", resources)
    for row in payload.get("requirement_candidates", []):
        require("requirement_candidates", row, "requirement_id", requirements)
        require("requirement_candidates", row, "resource_id", resources)
    for row in payload.get("plan_experiments", []):
        require("plan_experiments", row, "execution_plan_id", plans)
        require("plan_experiments", row, "experiment_spec_id", experiments)
    for row in payload.get("resource_bindings", []):
        require("resource_bindings", row, "plan_experiment_id", plan_experiments)
        require("resource_bindings", row, "requirement_id", requirements)
        require("resource_bindings", row, "resource_version_id", versions)
        plan_experiment = plan_experiments.get(row.get("plan_experiment_id"))
        requirement = requirements.get(row.get("requirement_id"))
        if (
            plan_experiment
            and requirement
            and plan_experiment["experiment_spec_id"]
            != requirement["experiment_spec_id"]
        ):
            raise ValueError(
                "resource binding joins different experiment specifications: "
                f"{row['plan_experiment_id']!r} and {row['requirement_id']!r}"
            )
    for row in payload.get("planned_run_slots", []):
        require("planned_run_slots", row, "plan_experiment_id", plan_experiments)

    bindings = {
        (row["plan_experiment_id"], row["requirement_id"]): row
        for row in payload.get("resource_bindings", [])
    }
    slots_by_experiment: dict[str, list[dict[str, Any]]] = {}
    for row in payload.get("planned_run_slots", []):
        slots_by_experiment.setdefault(row["plan_experiment_id"], []).append(row)
    for plan_experiment_id in slots_by_experiment:
        plan_experiment = plan_experiments[plan_experiment_id]
        plan = plans[plan_experiment["execution_plan_id"]]
        if not plan_experiment.get("enabled", True) or plan.get("status") != "runnable":
            raise ValueError(
                f"planned slots require an enabled runnable plan experiment: "
                f"{plan_experiment_id!r}"
            )
        required = [
            row
            for row in payload.get("requirements", [])
            if row["experiment_spec_id"] == plan_experiment["experiment_spec_id"]
            and row.get("required", True)
        ]
        for requirement in required:
            binding = bindings.get((plan_experiment_id, requirement["requirement_id"]))
            if binding is None:
                raise ValueError(
                    f"planned slots require binding {requirement['requirement_id']!r}"
                )
            version = versions[binding["resource_version_id"]]
            runtime_corpus = requirement.get("role") == "train_data"
            if not version.get("is_verified") or (
                not runtime_corpus and not version.get("checksum")
            ):
                raise ValueError(
                    "planned slots require a verified immutable resource version: "
                    f"{version['resource_version_id']!r}"
                )


def read_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("catalog manifest must be a JSON object")
    if payload.get("schema_version") != 1:
        raise ValueError("catalog manifest schema_version must be 1")
    unknown = set(payload) - (
        {"schema_version", "planned_run_matrices"} | set(SECTIONS)
    )
    if unknown:
        raise ValueError(f"unknown catalog manifest sections: {sorted(unknown)}")
    _expand_planned_run_matrices(payload)
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
    _validate_references(payload)
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
        "requirements": repo.upsert_requirement,
        "requirement_candidates": repo.upsert_requirement_candidate,
        "execution_plans": repo.upsert_execution_plan,
        "plan_experiments": repo.upsert_plan_experiment,
        "resource_bindings": repo.bind_plan_requirement,
        "planned_run_slots": repo.upsert_planned_run_slot,
    }
    counts: dict[str, int] = {}
    for section in SECTIONS:
        rows = payload.get(section, [])
        if section == "resources":
            for row in rows:
                repo.upsert_resource(**(row | {"canonical_version_id": None}))
        elif section == "resource_versions":
            for row in rows:
                repo.upsert_resource_version(**row)
            for resource in payload.get("resources", []):
                if canonical_version_id := resource.get("canonical_version_id"):
                    repo.set_resource_canonical_version(
                        resource["resource_id"], canonical_version_id
                    )
        else:
            for row in rows:
                calls[section](**row)
        counts[section] = len(rows)
    return counts


__all__ = ["digest_manifest", "load_manifest", "read_manifest"]
