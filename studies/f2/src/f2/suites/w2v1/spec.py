"""Validated W2V1 run specification."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from repro_core.execution.spec import load_variant, mapping


def _digest(value: dict[str, object]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class RunSpec:
    atomic_run_id: str
    identity: dict[str, object]
    corpus: dict[str, object]
    vocabulary: dict[str, object]
    training: dict[str, object]
    checkpoint: dict[str, object]
    tracking: dict[str, object]
    path: Path

    def with_seed(self, seed: int) -> RunSpec:
        """Bind a planner-selected seed to its immutable catalog slot."""
        identity = dict(self.identity)
        identity["seed"] = seed
        identity["planned_run_slot_id"] = (
            f"{identity['execution_plan_id']}-"
            f"{_condition_id(self.atomic_run_id)}-s{seed}"
        )
        return RunSpec(
            atomic_run_id=self.atomic_run_id,
            identity=identity,
            corpus=self.corpus,
            vocabulary=self.vocabulary,
            training=self.training,
            checkpoint=self.checkpoint,
            tracking=self.tracking,
            path=self.path,
        )

    def to_executor_config(self) -> dict[str, object]:
        resolved = {
            "corpus": self.corpus,
            "vocabulary": self.vocabulary,
            "training": self.training,
        }
        return {
            "kind": "word2vec",
            "atomic_run_id": self.atomic_run_id,
            "identity": {**self.identity, "config_digest": _digest(resolved)},
            **resolved,
            "checkpoint": self.checkpoint,
            "tracking": self.tracking,
        }


def parse_run_spec(
    path: str | Path,
    *,
    atomic_run_id: str | None = None,
    overrides: dict[str, object] | None = None,
) -> RunSpec:
    path = Path(path)
    raw = load_variant(path, atomic_run_id=atomic_run_id, overrides=overrides)
    if raw.get("domain") != "f2.w2v1" or raw.get("kind") != "word2vec":
        raise ValueError("W2V1 config requires domain f2.w2v1 and kind word2vec")
    identity = mapping(raw, "identity")
    required = {
        "execution_plan_id",
        "planned_run_slot_id",
    }
    if missing := sorted(required - identity.keys()):
        raise ValueError(f"W2V1 identity is missing: {', '.join(missing)}")
    training = mapping(raw, "training")
    return RunSpec(
        atomic_run_id=str(raw["atomic_run_id"]),
        identity=identity,
        corpus=mapping(raw, "corpus"),
        vocabulary=mapping(raw, "vocabulary"),
        training=training,
        checkpoint=mapping(raw, "checkpoint"),
        tracking=mapping(raw, "tracking"),
        path=path,
    )


def _condition_id(atomic_run_id: str) -> str:
    try:
        corpus_id, condition_id = atomic_run_id.split("--", 1)
    except ValueError as exc:
        raise ValueError(
            "canonical W2V atomic run IDs must be <corpus>--<condition>"
        ) from exc
    if not corpus_id or not condition_id:
        raise ValueError("W2V atomic run corpus and condition IDs must be non-empty")
    return condition_id


__all__ = ["RunSpec", "parse_run_spec"]
