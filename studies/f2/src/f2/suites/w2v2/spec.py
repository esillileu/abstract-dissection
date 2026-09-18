"""Validated W2V2 phrase run specification."""

from __future__ import annotations

from pathlib import Path

from f2.suites.w2v1.spec import RunSpec as CommonRunSpec
from f2.suites.w2v1.spec import _digest
from repro_core.execution.spec import load_variant, mapping


class RunSpec(CommonRunSpec):
    def to_executor_config(self) -> dict[str, object]:
        config = super().to_executor_config()
        config["phrase_detection"] = self.phrase_detection
        resolved = {
            key: config[key]
            for key in (
                "corpus",
                "vocabulary",
                "training",
                "evaluation",
                "phrase_detection",
            )
        }
        config["identity"] = {**self.identity, "config_digest": _digest(resolved)}
        return config

    @property
    def phrase_detection(self) -> dict[str, object]:
        return self._phrase_detection


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
        "resource_version",
        "corpus_manifest_digest",
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
    spec = RunSpec(
        atomic_run_id=str(raw["atomic_run_id"]),
        identity=identity,
        corpus=mapping(raw, "corpus"),
        vocabulary=mapping(raw, "vocabulary"),
        training=training,
        evaluation=mapping(raw, "evaluation"),
        checkpoint=mapping(raw, "checkpoint"),
        tracking=mapping(raw, "tracking"),
        path=path,
    )
    object.__setattr__(spec, "_phrase_detection", phrase)
    return spec


__all__ = ["RunSpec", "parse_run_spec"]
