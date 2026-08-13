"""Small RunSpec contract shared by promoted upstream experiment domains."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from exp.framework.execution.spec import load_variant


@dataclass(frozen=True)
class PromotedRunSpec:
    config: dict[str, object]

    def to_executor_config(self) -> dict[str, object]:
        return dict(self.config)


def parse(path: str | Path, *, domain: str, atomic_run_id: str | None, overrides: dict[str, object] | None) -> PromotedRunSpec:
    path = Path(path)
    raw = load_variant(path, atomic_run_id=atomic_run_id, overrides=overrides)
    if raw.get("domain") != domain:
        raise ValueError(f"{domain} YAML requires domain: {domain}: {path}")
    run = raw.get("run")
    if not isinstance(run, dict):
        raise ValueError(f"run must be a mapping: {path}")
    required = ("trial_id", "source_experiment")
    missing = [key for key in required if not raw.get(key)]
    if missing:
        raise ValueError(f"missing promoted original fields {missing}: {path}")
    config = {
        "schema_version": 1,
        "kind": domain,
        "atomic_run_id": str(raw["atomic_run_id"]),
        "experiment_ids": [str(run["experiment_id"])],
        "execution_group_id": str(run.get("group_id", run["experiment_id"])),
        "recipe_id": str(run.get("recipe_id", raw["trial_id"])),
        "protocol_version": str(run.get("protocol_version", "book-source-v1")),
        "structure_signature": str(run.get("structure_signature", raw["trial_id"])),
        "trial_id": str(raw["trial_id"]),
        "source_experiment": str(raw["source_experiment"]),
        "conditions": dict(raw.get("conditions", {})),
        "dataset": dict(raw.get("dataset", {})),
        "model": dict(raw.get("model", {})),
        "optimizer": dict(raw.get("optimizer", {})),
        "loader": dict(raw.get("loader", {})),
        "training": {"entrypoint": str(path), **dict(raw.get("budget", {}))},
        "evaluation": dict(raw.get("evaluation", {})),
        "numerics": dict(raw.get("numerics", {})),
        "checkpoint": dict(raw.get("checkpoint", {})),
        "profiling": dict(raw.get("profiling", {})),
        "policy": {
            "dataset_split_seed": 1984,
            **dict(raw.get("seed_policy", {})),
        },
        "tracking": dict(raw.get("tracking", {})),
    }
    return PromotedRunSpec(config)
