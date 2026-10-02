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
    distribution: dict[str, object] | None = None

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
            distribution=self.distribution,
        )

    def to_executor_config(self) -> dict[str, object]:
        resolved: dict[str, object] = {
            "corpus": self.corpus,
            "vocabulary": self.vocabulary,
            "training": self.training,
        }
        if self.distribution is not None:
            resolved["distribution"] = self.distribution
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
    distribution = mapping(raw, "distribution") if "distribution" in raw else None
    if "study" not in identity:
        if "table6" in path.name:
            identity["study"] = "table6"
        elif "table3" in path.name:
            identity["study"] = "table3"
        elif "table4" in path.name:
            identity["study"] = "table4"
        elif "table5" in path.name:
            identity["study"] = "table5"
        else:
            identity["study"] = "table2"
    if "experiment_spec_id" not in identity:
        if raw.get("experiment_spec_id"):
            identity["experiment_spec_id"] = raw["experiment_spec_id"]
        elif identity.get("study") == "table6" or "table6" in path.name:
            model_kind = training.get("model_kind")
            if model_kind == "cbow":
                identity["experiment_spec_id"] = "w2v1-table6-cbow-6b"
            elif model_kind == "skip_gram":
                identity["experiment_spec_id"] = "w2v1-table6-skipgram-6b"
            else:
                identity["experiment_spec_id"] = "w2v1-table6"
        elif identity.get("study") == "table3" or "table3" in path.name:
            model_kind = training.get("model_kind")
            if model_kind == "cbow":
                identity["experiment_spec_id"] = "w2v1-table3-cbow"
            elif model_kind == "skip_gram":
                identity["experiment_spec_id"] = "w2v1-table3-skipgram"
            else:
                identity["experiment_spec_id"] = "w2v1-table3"
        else:
            identity["experiment_spec_id"] = "w2v1-table2-cbow"
    return RunSpec(
        atomic_run_id=str(raw["atomic_run_id"]),
        identity=identity,
        corpus=mapping(raw, "corpus"),
        vocabulary=mapping(raw, "vocabulary"),
        training=training,
        checkpoint=mapping(raw, "checkpoint"),
        tracking=mapping(raw, "tracking"),
        path=path,
        distribution=distribution,
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
