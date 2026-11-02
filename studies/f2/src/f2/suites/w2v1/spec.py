"""Validated W2V1 run specification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from f2.suites.w2v.spec import Word2VecRunSpec, _condition_id, _digest
from repro_core.execution.spec import load_variant, mapping


@dataclass(frozen=True)
class RunSpec(Word2VecRunSpec):
    """W2V1 paper reproduction run specification inheriting top-down defaults."""


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
        if "table7" in path.name:
            identity["study"] = "table7"
        elif "table6" in path.name:
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
        elif identity.get("study") == "table7" or "table7" in path.name:
            identity["experiment_spec_id"] = "w2v1-msr-sentence-skipgram"
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


__all__ = ["RunSpec", "_condition_id", "_digest", "parse_run_spec"]
