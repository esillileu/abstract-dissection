"""Base validated Word2Vec run specification and top-down configuration resolution."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Self


def _digest(value: dict[str, object]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


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


@dataclass(frozen=True)
class Word2VecRunSpec:
    """Top-level configuration specification for Word2Vec experiments.

    Provides top-down defaults and hierarchical overrides for seed binding
    and executor configuration generation.
    """

    atomic_run_id: str
    identity: dict[str, object]
    corpus: dict[str, object]
    vocabulary: dict[str, object]
    training: dict[str, object]
    checkpoint: dict[str, object]
    tracking: dict[str, object]
    path: Path
    distribution: dict[str, object] | None = None

    def with_seed(self: Self, seed: int) -> Self:
        """Bind a planner-selected seed to its immutable catalog slot."""
        identity = dict(self.identity)
        identity["seed"] = seed
        identity["planned_run_slot_id"] = (
            f"{identity['execution_plan_id']}-"
            f"{_condition_id(self.atomic_run_id)}-s{seed}"
        )
        return dataclasses.replace(self, identity=identity)

    def resolved_config(self) -> dict[str, object]:
        """Resolve semantic configuration mappings for identity digest computation."""
        resolved: dict[str, object] = {
            "corpus": self.corpus,
            "vocabulary": self.vocabulary,
            "training": self.training,
        }
        if self.distribution is not None:
            resolved["distribution"] = self.distribution
        return resolved

    def to_executor_config(self) -> dict[str, object]:
        """Convert specification to engine executor configuration."""
        resolved = self.resolved_config()
        return {
            "kind": "word2vec",
            "atomic_run_id": self.atomic_run_id,
            "identity": {**self.identity, "config_digest": _digest(resolved)},
            **resolved,
            "checkpoint": self.checkpoint,
            "tracking": self.tracking,
        }


__all__ = [
    "Word2VecRunSpec",
    "_condition_id",
    "_digest",
]
