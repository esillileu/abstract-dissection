"""The portable, YAML-facing experiment configuration contract.

This module deliberately contains no tracking implementation.  Integrations may
use the normalized sections to persist their own representation.
"""

from __future__ import annotations

from copy import deepcopy
_SECTIONS = (
    "dataset", "loader", "model", "initializer", "optimizer", "scheduler",
    "loss", "training", "evaluation", "numerics", "checkpoint", "profiling",
    "policy", "regularization",
)


def normalize_config(config: dict[str, object]) -> dict[str, object]:
    """Return a self-contained experiment configuration with stable defaults."""
    if not isinstance(config.get("kind"), str) or not config["kind"]:
        raise ValueError("experiment config requires a non-empty string 'kind'")

    value: dict[str, object] = deepcopy(config)
    metadata = _mapping(value, "metadata")
    value["schema_version"] = int(value.get("schema_version", 1))
    if value["schema_version"] != 1:
        raise ValueError("only experiment schema_version: 1 is supported")
    value["atomic_run_id"] = str(value.get("atomic_run_id", metadata.get("atomic_run_id", value["kind"])))
    experiment_ids = value.get("experiment_ids", metadata.get("experiment_ids", (value["kind"],)))
    if isinstance(experiment_ids, str):
        experiment_ids = [experiment_ids]
    if not isinstance(experiment_ids, list | tuple) or not experiment_ids:
        raise ValueError("experiment_ids must be a non-empty list")
    value["experiment_ids"] = [str(item) for item in experiment_ids]
    value["execution_group_id"] = str(value.get("execution_group_id", metadata.get("execution_group_id", value["kind"])))
    value["recipe_id"] = str(value.get("recipe_id", metadata.get("recipe_id", value["kind"])))
    value["structure_signature"] = str(value.get("structure_signature", metadata.get("structure_signature", value["kind"])))
    value["seed"] = int(value.get("seed", 0))

    for section in _SECTIONS:
        value[section] = _mapping(value, section)
    training = _mapping(value, "training")
    training.setdefault("entrypoint", str(value.get("entrypoint", f"yaml:{value['kind']}")))
    value["training"] = training
    value["run"] = _mapping(value, "run")
    value["tracking"] = _mapping(value, "tracking")
    value.pop("metadata", None)
    return value


def _mapping(config: dict[str, object], key: str) -> dict[str, object]:
    raw = config.get(key, {})
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"{key} must be a mapping")
    return dict(raw)
