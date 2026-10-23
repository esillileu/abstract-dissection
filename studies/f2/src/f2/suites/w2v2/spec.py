"""Validated W2V2 phrase run specification."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from f2.suites.w2v.spec import Word2VecRunSpec
from repro_core.execution.spec import load_variant, mapping


@dataclass(frozen=True)
class RunSpec(Word2VecRunSpec):
    """W2V2 phrase specification extending Word2VecRunSpec with phrase detection."""

    phrase_detection: dict[str, object] = field(default_factory=dict)

    def resolved_config(self) -> dict[str, object]:
        resolved = super().resolved_config()
        resolved["phrase_detection"] = self.phrase_detection
        return resolved


def parse_run_spec(
    path: str | Path,
    *,
    atomic_run_id: str | None = None,
    overrides: dict[str, object] | None = None,
) -> RunSpec:
    path = Path(path)
    raw = load_variant(path, atomic_run_id=atomic_run_id, overrides=overrides)
    if raw.get("domain") != "f2.w2v2" or raw.get("kind") != "word2vec":
        raise ValueError("W2V2 config requires domain f2.w2v2 and kind word2vec")
    identity = mapping(raw, "identity")
    required = {
        "execution_plan_id",
        "planned_run_slot_id",
        "seed",
    }
    if missing := sorted(required - identity.keys()):
        raise ValueError(f"W2V2 identity is missing: {', '.join(missing)}")
    training = mapping(raw, "training")
    objective = training.get("objective_kind")
    if objective not in {"negative_sampling", "hierarchical_softmax"}:
        raise ValueError(
            f"unsupported W2V2 objective: {objective}; NCE is not substituted with NEG"
        )
    phrase = mapping(raw, "phrase_detection")
    return RunSpec(
        atomic_run_id=str(raw["atomic_run_id"]),
        identity=identity,
        corpus=mapping(raw, "corpus"),
        vocabulary=mapping(raw, "vocabulary"),
        training=training,
        checkpoint=mapping(raw, "checkpoint"),
        tracking=mapping(raw, "tracking"),
        path=path,
        phrase_detection=phrase,
    )


__all__ = ["RunSpec", "parse_run_spec"]
