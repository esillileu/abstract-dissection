"""YAML parser and validation rules for DS2 RunSpec."""

from __future__ import annotations

from pathlib import Path

from repro_core.execution.spec import RunIdentity, load_variant, mapping

from .types import EvaluationTrigger, RunSpec, SourceCurveSpec


def _source_curve(recording: dict[str, object]) -> SourceCurveSpec | None:
    raw = recording.get("source_curve")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("recording.source_curve must be a mapping")
    return SourceCurveSpec(
        kind=str(raw["kind"]),
        every_updates=None
        if raw.get("every_updates") is None
        else int(raw["every_updates"]),
        every_epochs=None
        if raw.get("every_epochs") is None
        else int(raw["every_epochs"]),
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
        output.append(
            EvaluationTrigger(
                axis=str(item["axis"]),  # type: ignore[arg-type]
                sources=tuple(str(source) for source in sources),
                every=None if item.get("every") is None else int(item["every"]),
            )
        )
    return tuple(output)


def _validate(spec: RunSpec) -> None:
    if spec.identity.group_id.startswith("GO") and spec.kind != "observation":
        raise ValueError("DS2 GO groups must use kind: observation")
    if spec.identity.group_id.startswith("GT") and spec.kind == "observation":
        raise ValueError("DS2 GT groups must not use kind: observation")
    if spec.identity.group_id.startswith("PF") and spec.kind != "performance_profile":
        raise ValueError("DS2 PF groups must use kind: performance_profile")
    if spec.kind == "performance_profile" and not spec.identity.group_id.startswith(
        "PF"
    ):
        raise ValueError("DS2 performance profiles must use a PF group")
    if spec.source_curve is not None:
        if (
            spec.source_curve.every_updates is not None
            and spec.source_curve.every_updates < 1
        ):
            raise ValueError("source_curve.every_updates must be positive")
        if (
            spec.source_curve.every_epochs is not None
            and spec.source_curve.every_epochs < 1
        ):
            raise ValueError("source_curve.every_epochs must be positive")


def _reject_old_catalog_keys(raw: dict[str, object]) -> None:
    old_keys = sorted({"training", "evaluation", "policy"} & set(raw))
    if old_keys:
        raise ValueError(
            "training/evaluation/policy must be replaced by RunSpec budget/recording/seed_policy fields; "
            f"old catalog keys are not supported: {old_keys}"
        )


def parse_run_spec(
    path: str | Path,
    *,
    atomic_run_id: str | None = None,
    overrides: dict[str, object] | None = None,
) -> RunSpec:
    path = Path(path)
    raw = load_variant(path, atomic_run_id=atomic_run_id, overrides=overrides)
    if raw.get("domain") != "deepscratch.ds2.implemented":
        raise ValueError(
            f"DS2 implemented YAML requires domain: deepscratch.ds2.implemented: {path}"
        )
    if raw.get("kind") not in {
        "word2vec",
        "language_modeling",
        "seq2seq",
        "observation",
        "performance_profile",
        "count_based_embedding",
    }:
        raise ValueError(f"DS2 does not support kind: {raw.get('kind')}")
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
        recording=recording,
        source_curve=_source_curve(recording),
        evaluations=_evaluations(recording),
        checkpoint=mapping(raw, "checkpoint"),
        tracking=mapping(raw, "tracking"),
        numerics=mapping(raw, "numerics"),
        profiling=mapping(raw, "profiling"),
        scheduler=mapping(raw, "scheduler"),
        objective=mapping(raw, "objective"),
        path=path,
        protocol_version=str(run.get("protocol_version", "legacy")),
    )
    _validate(spec)
    return spec
