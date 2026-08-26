"""DS1 RunSpec: the code contract that connects docs to YAML."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from repro_core.execution.spec import RunIdentity, load_variant, mapping

TriggerType = Literal["updates", "epoch_first_update", "epoch_end", "terminal"]


@dataclass(frozen=True)
class EvaluationSource:
    id: str
    split: Literal["train", "valid", "test"]
    kind: Literal["first_n", "full"]
    count: int | None = None


@dataclass(frozen=True)
class Trigger:
    type: TriggerType
    sources: tuple[str, ...]
    start: int | None = None
    every: int | None = None
    stop: int | None = None


@dataclass(frozen=True)
class RunSpec:
    kind: Literal["supervised_classification", "observation"]
    identity: RunIdentity
    atomic_run_id: str
    seed_policy: dict[str, object]
    dataset: dict[str, object]
    model: dict[str, object]
    optimizer: dict[str, object]
    loader: dict[str, object]
    budget: dict[str, object]
    evaluation_sources: tuple[EvaluationSource, ...]
    triggers: tuple[Trigger, ...]
    checkpoint: dict[str, object]
    tracking: dict[str, object]
    numerics: dict[str, object]
    profiling: dict[str, object]
    objective: dict[str, object]
    path: Path
    protocol_version: str = "legacy"

    @property
    def config(self) -> dict[str, object]:
        return self.to_executor_config()

    def to_executor_config(self) -> dict[str, object]:
        schedule: dict[str, object] = {}
        for trigger in self.triggers:
            payload: dict[str, object] = {"sets": list(trigger.sources)}
            if trigger.type == "updates":
                if trigger.start is None or trigger.every is None:
                    raise ValueError("updates trigger requires start and every")
                payload.update({"start": trigger.start, "every": trigger.every})
                if trigger.stop is not None:
                    payload["stop"] = trigger.stop
                schedule["on_update"] = payload
            elif trigger.type == "epoch_first_update":
                schedule["on_epoch_first_update"] = payload
            elif trigger.type == "epoch_end":
                schedule["on_epoch_end"] = payload
            elif trigger.type == "terminal":
                schedule["on_train_end"] = payload
            else:
                raise ValueError(f"unsupported DS1 trigger: {trigger.type}")
        return {
            "kind": self.kind,
            "atomic_run_id": self.atomic_run_id,
            "experiment_ids": [self.identity.experiment_id],
            "execution_group_id": self.identity.group_id,
            "recipe_id": self.identity.recipe_id,
            "protocol_version": self.protocol_version,
            "structure_signature": self.identity.structure_signature,
            "dataset": dict(self.dataset),
            "loader": dict(self.loader),
            "model": dict(self.model),
            "optimizer": dict(self.optimizer),
            "objective": dict(self.objective),
            "training": {
                "entrypoint": str(self.path),
                "max_epochs": self.budget.get("max_epochs"),
                **(
                    {"max_updates": self.budget["max_updates"]}
                    if "max_updates" in self.budget
                    else {}
                ),
            },
            "evaluation": {
                "primary_metric": "test/accuracy",
                "sources": [source.__dict__ for source in self.evaluation_sources],
                "schedule": schedule,
            },
            "numerics": dict(self.numerics),
            "checkpoint": dict(self.checkpoint),
            "profiling": dict(self.profiling),
            "policy": {
                "seed_set": self.seed_policy.get("seed_set", "research_v1"),
                "seed_count": self.seed_policy.get("seed_count", 10),
                "paired_execution": self.seed_policy.get("paired_execution", True),
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
    raw = load_variant(path, atomic_run_id=atomic_run_id, overrides=overrides)
    if raw.get("domain") != "deepscratch.ds1.implemented":
        raise ValueError(
            f"DS1 implemented YAML requires domain: deepscratch.ds1.implemented: {path}"
        )
    if raw.get("kind") not in {"supervised_classification", "observation"}:
        raise ValueError(f"DS1 does not support kind: {raw.get('kind')}")
    _reject_old_catalog_keys(raw)
    run = mapping(raw, "run")
    recording = mapping(raw, "recording")
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
        seed_policy=mapping(raw, "seed_policy"),
        dataset=mapping(raw, "dataset"),
        model=mapping(raw, "model"),
        optimizer=mapping(raw, "optimizer"),
        loader=mapping(raw, "loader"),
        budget=mapping(raw, "budget"),
        evaluation_sources=_sources(recording),
        triggers=_triggers(recording),
        checkpoint=mapping(raw, "checkpoint"),
        tracking=mapping(raw, "tracking"),
        numerics=mapping(raw, "numerics"),
        profiling=mapping(raw, "profiling"),
        objective=mapping(raw, "objective"),
        path=path,
        protocol_version=str(run.get("protocol_version", "legacy")),
    )
    _validate(spec)
    return spec


def _sources(recording: dict[str, object]) -> tuple[EvaluationSource, ...]:
    raw = recording.get("evaluation_sources", ())
    if not isinstance(raw, list | tuple):
        raise ValueError("recording.evaluation_sources must be a list")
    sources = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("evaluation source must be a mapping")
        sources.append(
            EvaluationSource(
                id=str(item["id"]),
                split=str(item["split"]),  # type: ignore[arg-type]
                kind=str(item["kind"]),  # type: ignore[arg-type]
                count=None if item.get("count") is None else int(item["count"]),
            )
        )
    return tuple(sources)


def _triggers(recording: dict[str, object]) -> tuple[Trigger, ...]:
    raw = recording.get("triggers", ())
    if not isinstance(raw, list | tuple):
        raise ValueError("recording.triggers must be a list")
    triggers = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("trigger must be a mapping")
        sources = item.get("sources", ())
        if not isinstance(sources, list | tuple):
            raise ValueError("trigger.sources must be a list")
        triggers.append(
            Trigger(
                type=str(item["type"]),  # type: ignore[arg-type]
                sources=tuple(str(source) for source in sources),
                start=None if item.get("start") is None else int(item["start"]),
                every=None if item.get("every") is None else int(item["every"]),
                stop=None if item.get("stop") is None else int(item["stop"]),
            )
        )
    return tuple(triggers)


def _validate(spec: RunSpec) -> None:
    source_ids = {source.id for source in spec.evaluation_sources}
    for trigger in spec.triggers:
        missing = sorted(set(trigger.sources) - source_ids)
        if missing:
            raise ValueError(
                f"trigger references unknown evaluation sources: {missing}"
            )
        if trigger.type == "updates":
            if trigger.start is None or trigger.every is None:
                raise ValueError("updates trigger requires start and every")
            if trigger.start < 1 or trigger.every < 1:
                raise ValueError("updates trigger start/every must be positive")
    if spec.identity.group_id.startswith("GO") and spec.kind != "observation":
        raise ValueError("DS1 GO groups must use kind: observation")
    if (
        spec.identity.group_id.startswith("GT")
        and spec.kind != "supervised_classification"
    ):
        raise ValueError("DS1 GT groups must use kind: supervised_classification")


def _reject_old_catalog_keys(raw: dict[str, object]) -> None:
    old_keys = sorted({"training", "evaluation", "policy"} & set(raw))
    if old_keys:
        raise ValueError(
            "training must be replaced by RunSpec budget/recording fields; "
            f"old catalog keys are not supported: {old_keys}"
        )
