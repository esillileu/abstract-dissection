"""DS2 RunSpec: document-shaped contract for language and sequence runs."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml


@dataclass(frozen=True)
class RunIdentity:
    experiment_id: str
    group_id: str
    protocol: str
    recipe_id: str
    structure_signature: str


@dataclass(frozen=True)
class SourceCurveSpec:
    kind: str
    every_updates: int | None = None
    every_epochs: int | None = None
    reducer: str = "mean"
    plot_index: str = "zero_based_append"


@dataclass(frozen=True)
class EvaluationTrigger:
    axis: Literal["epoch", "terminal"]
    sources: tuple[str, ...]
    every: int | None = None


@dataclass(frozen=True)
class RunSpec:
    kind: Literal["word2vec", "language_modeling", "seq2seq", "observation"]
    identity: RunIdentity
    atomic_run_id: str
    seed_policy: dict[str, object]
    dataset: dict[str, object]
    model: dict[str, object]
    optimizer: dict[str, object]
    loader: dict[str, object]
    budget: dict[str, object]
    recording: dict[str, object]
    source_curve: SourceCurveSpec | None
    evaluations: tuple[EvaluationTrigger, ...]
    checkpoint: dict[str, object]
    tracking: dict[str, object]
    numerics: dict[str, object]
    profiling: dict[str, object]
    scheduler: dict[str, object]
    objective: dict[str, object]
    path: Path

    @property
    def config(self) -> dict[str, object]:
        return self.to_executor_config()

    def to_executor_config(self) -> dict[str, object]:
        evaluation: dict[str, object] = {"primary_metric": "final/train/loss"}
        for trigger in self.evaluations:
            if trigger.axis == "epoch":
                evaluation["valid_every_epochs"] = trigger.every if "valid" in trigger.sources else 0
                evaluation["test_every_epochs"] = trigger.every if "test" in trigger.sources else 0
            elif trigger.axis == "terminal":
                evaluation["test_at_end"] = "test" in trigger.sources
        if self.kind == "seq2seq":
            evaluation.update({"primary_metric": "final/test/exact_match", "decode": "greedy", "test_every_epochs": 1})
        elif self.kind == "language_modeling":
            evaluation.setdefault("primary_metric", "final/test/perplexity")
            evaluation.setdefault("valid_every_epochs", 0)
            evaluation.setdefault("test_every_epochs", 0)
            evaluation.setdefault("test_at_end", False)
        recording = dict(self.recording)
        if self.source_curve is not None:
            recording["source_curve"] = {
                "kind": self.source_curve.kind,
                **({} if self.source_curve.every_updates is None else {"every_updates": self.source_curve.every_updates}),
                **({} if self.source_curve.every_epochs is None else {"every_epochs": self.source_curve.every_epochs}),
                "reducer": self.source_curve.reducer,
                "plot_index": self.source_curve.plot_index,
            }
        return {
            "kind": self.kind,
            "atomic_run_id": self.atomic_run_id,
            "experiment_ids": [self.identity.experiment_id],
            "execution_group_id": self.identity.group_id,
            "recipe_id": self.identity.recipe_id,
            "structure_signature": self.identity.structure_signature,
            "dataset": dict(self.dataset),
            "loader": dict(self.loader),
            "model": dict(self.model),
            "optimizer": dict(self.optimizer),
            "scheduler": dict(self.scheduler),
            "objective": dict(self.objective),
            "training": {
                "entrypoint": str(self.path),
                "max_epochs": self.budget.get("max_epochs"),
                **({"max_updates": self.budget["max_updates"]} if "max_updates" in self.budget else {}),
                **({"loop": self.budget["loop"]} if "loop" in self.budget else {}),
            },
            "recording": recording,
            "evaluation": evaluation,
            "numerics": dict(self.numerics),
            "checkpoint": dict(self.checkpoint),
            "profiling": dict(self.profiling),
            "policy": {
                "seed_set": self.seed_policy.get("seed_set", "research_v1"),
                "seed_count": self.seed_policy.get("seed_count", 10),
                "paired_execution": self.seed_policy.get("paired_execution", True),
                **({"max_grad": self.seed_policy["max_grad"]} if "max_grad" in self.seed_policy else {}),
            },
            "tracking": dict(self.tracking),
        }


def parse_run_spec(
    path: str | Path,
    *,
    atomic_run_id: str | None = None,
    overrides: dict[str, object] | None = None,
) -> RunSpec:
    path = Path(path)
    raw = _load_variant(path, atomic_run_id=atomic_run_id, overrides=overrides)
    if raw.get("domain") != "ds2":
        raise ValueError(f"DS2 YAML requires domain: ds2: {path}")
    if raw.get("kind") not in {"word2vec", "language_modeling", "seq2seq", "observation"}:
        raise ValueError(f"DS2 does not support kind: {raw.get('kind')}")
    _reject_old_catalog_keys(raw)
    run = _mapping(raw, "run")
    recording = _mapping(raw, "recording")
    spec = RunSpec(
        kind=str(raw["kind"]),  # type: ignore[arg-type]
        identity=RunIdentity(
            experiment_id=str(run["experiment_id"]),
            group_id=str(run["group_id"]),
            protocol=str(run["protocol"]),
            recipe_id=str(run["recipe_id"]),
            structure_signature=str(run["structure_signature"]),
        ),
        atomic_run_id=str(raw["atomic_run_id"]),
        seed_policy=_mapping(raw, "seed_policy"),
        dataset=_mapping(raw, "dataset"),
        model=_mapping(raw, "model"),
        optimizer=_mapping(raw, "optimizer"),
        loader=_mapping(raw, "loader"),
        budget=_mapping(raw, "budget"),
        recording=recording,
        source_curve=_source_curve(recording),
        evaluations=_evaluations(recording),
        checkpoint=_mapping(raw, "checkpoint"),
        tracking=_mapping(raw, "tracking"),
        numerics=_mapping(raw, "numerics"),
        profiling=_mapping(raw, "profiling"),
        scheduler=_mapping(raw, "scheduler"),
        objective=_mapping(raw, "objective"),
        path=path,
    )
    _validate(spec)
    return spec


def _load_variant(path: Path, *, atomic_run_id: str | None, overrides: dict[str, object] | None) -> dict[str, object]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"invalid YAML object: {path}")
    variants = value.pop("variants", None)
    if not isinstance(variants, dict) or not variants:
        raise ValueError(f"RunSpec YAML needs variants: {path}")
    if atomic_run_id is None:
        raise ValueError("this YAML defines variants; choose --atomic-run-id")
    selected = variants.get(atomic_run_id)
    if not isinstance(selected, dict):
        raise ValueError(f"unknown atomic_run_id: {atomic_run_id}")
    value = _merge(value, selected)
    value["atomic_run_id"] = atomic_run_id
    if overrides:
        value = _merge(value, overrides)
    return value


def _source_curve(recording: dict[str, object]) -> SourceCurveSpec | None:
    raw = recording.get("source_curve")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("recording.source_curve must be a mapping")
    return SourceCurveSpec(
        kind=str(raw["kind"]),
        every_updates=None if raw.get("every_updates") is None else int(raw["every_updates"]),
        every_epochs=None if raw.get("every_epochs") is None else int(raw["every_epochs"]),
        reducer=str(raw.get("reducer", "mean")),
        plot_index=str(raw.get("plot_index", "zero_based_append")),
    )


def _evaluations(recording: dict[str, object]) -> tuple[EvaluationTrigger, ...]:
    raw = recording.get("evaluations", ())
    if not isinstance(raw, list | tuple):
        raise ValueError("recording.evaluations must be a list")
    output = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("evaluation trigger must be a mapping")
        sources = item.get("sources", ())
        if not isinstance(sources, list | tuple):
            raise ValueError("evaluation trigger sources must be a list")
        output.append(EvaluationTrigger(
            axis=str(item["axis"]),  # type: ignore[arg-type]
            sources=tuple(str(source) for source in sources),
            every=None if item.get("every") is None else int(item["every"]),
        ))
    return tuple(output)


def _validate(spec: RunSpec) -> None:
    if spec.identity.group_id.startswith("GO") and spec.kind != "observation":
        raise ValueError("DS2 GO groups must use kind: observation")
    if spec.identity.group_id.startswith("GT") and spec.kind == "observation":
        raise ValueError("DS2 GT groups must not use kind: observation")
    if spec.source_curve is not None:
        if spec.source_curve.every_updates is not None and spec.source_curve.every_updates < 1:
            raise ValueError("source_curve.every_updates must be positive")
        if spec.source_curve.every_epochs is not None and spec.source_curve.every_epochs < 1:
            raise ValueError("source_curve.every_epochs must be positive")


def _reject_old_catalog_keys(raw: dict[str, object]) -> None:
    old_keys = sorted({"training", "evaluation", "policy"} & set(raw))
    if old_keys:
        raise ValueError(
            "training/evaluation/policy must be replaced by RunSpec budget/recording/seed_policy fields; "
            f"old catalog keys are not supported: {old_keys}"
        )


def _mapping(config: dict[str, object], key: str) -> dict[str, object]:
    value = config.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping")
    return dict(value)


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, object]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result
