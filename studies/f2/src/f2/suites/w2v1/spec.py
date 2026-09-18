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
        from f2.suites.w2v.matrix import planned_slot_id

        identity = dict(self.identity)
        identity["seed"] = seed
        if self.atomic_run_id == "local-smoke":
            base = str(identity["planned_run_slot_id"]).rsplit("-s", 1)[0]
            identity["planned_run_slot_id"] = f"{base}-s{seed}"
        else:
            identity["planned_run_slot_id"] = planned_slot_id(
                str(identity["execution_plan_id"]), self.atomic_run_id, seed
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
        "resource_version",
        "corpus_manifest_digest",
    }
    if missing := sorted(required - identity.keys()):
        raise ValueError(f"W2V1 identity is missing: {', '.join(missing)}")
    training = mapping(raw, "training")
    if int(training.get("thread_count", 0)) != 1:
        raise ValueError("W2V1 parity configuration requires one training thread")
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


__all__ = ["RunSpec", "parse_run_spec"]
