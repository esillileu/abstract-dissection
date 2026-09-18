"""Transform DS2 RunSpec into executor configuration mappings."""

from __future__ import annotations

from typing import TYPE_CHECKING

from repro_core.execution.spec import mapping

if TYPE_CHECKING:
    from .types import RunSpec


def build_executor_config(spec: RunSpec) -> dict[str, object]:
    evaluation: dict[str, object] = {"primary_metric": "final/train/loss"}
    for trigger in spec.evaluations:
        if trigger.axis == "epoch":
            evaluation["valid_every_epochs"] = (
                trigger.every if "valid" in trigger.sources else 0
            )
            evaluation["test_every_epochs"] = (
                trigger.every if "test" in trigger.sources else 0
            )
        elif trigger.axis == "terminal":
            evaluation["test_at_end"] = "test" in trigger.sources
    if spec.kind == "seq2seq":
        evaluation.update(
            {"primary_metric": "final/test/exact_match", "decode": "greedy"}
        )
        evaluation.setdefault("test_every_epochs", 1)
    elif spec.kind == "language_modeling":
        evaluation.setdefault("primary_metric", "final/test/perplexity")
        evaluation.setdefault("valid_every_epochs", 0)
        evaluation.setdefault("test_every_epochs", 0)
        evaluation.setdefault("test_at_end", False)
        evaluation.setdefault("protocol", "book_parallel")
        evaluation.setdefault("batch_size", 10)
        evaluation.setdefault("time_size", 35)
        evaluation.setdefault("drop_remainder", True)
        evaluation.setdefault("terminal_checkpoint_role", "final")
    recording = dict(spec.recording)
    if spec.source_curve is not None:
        recording["source_curve"] = {
            "kind": spec.source_curve.kind,
            **(
                {}
                if spec.source_curve.every_updates is None
                else {"every_updates": spec.source_curve.every_updates}
            ),
            **(
                {}
                if spec.source_curve.every_epochs is None
                else {"every_epochs": spec.source_curve.every_epochs}
            ),
            "reducer": spec.source_curve.reducer,
            "plot_index": spec.source_curve.plot_index,
        }
    tracking = dict(spec.tracking)
    if spec.kind == "performance_profile":
        tags = dict(mapping(tracking, "tags"))
        tags.update(
            {
                "run.type": "profile",
                "profile.study": spec.identity.experiment_id,
                "profile.group": spec.identity.group_id,
                "profile.source_study": str(spec.dataset.get("source_study", "")),
                "profile.study_kind": str(spec.profiling.get("study_kind", "")),
                "profile.timing_source": str(
                    spec.profiling.get("timing_source", "window")
                ),
                "result.schema.name": "ds2-profile",
                "result.schema.version": "1",
            }
        )
        tracking["tags"] = tags
    return {
        "kind": spec.kind,
        "atomic_run_id": spec.atomic_run_id,
        "experiment_ids": [spec.identity.experiment_id],
        "execution_group_id": spec.identity.group_id,
        "recipe_id": spec.identity.recipe_id,
        "protocol_version": spec.protocol_version,
        "structure_signature": spec.identity.structure_signature,
        "dataset": dict(spec.dataset),
        "loader": dict(spec.loader),
        "model": dict(spec.model),
        "optimizer": dict(spec.optimizer),
        "scheduler": dict(spec.scheduler),
        "objective": dict(spec.objective),
        "loss": {"phase": spec.seed_policy.get("loss_timing", "post_update")},
        "training": {
            "entrypoint": str(spec.path),
            "max_epochs": spec.budget.get("max_epochs"),
            **(
                {"max_updates": spec.budget["max_updates"]}
                if "max_updates" in spec.budget
                else {}
            ),
            **({"loop": spec.budget["loop"]} if "loop" in spec.budget else {}),
        },
        "recording": recording,
        "evaluation": evaluation,
        "numerics": dict(spec.numerics),
        "checkpoint": dict(spec.checkpoint),
        "profiling": dict(spec.profiling),
        "policy": {
            "seed_set": spec.seed_policy.get("seed_set", "research_v1"),
            "seed_count": spec.seed_policy.get("seed_count", 10),
            "paired_execution": spec.seed_policy.get("paired_execution", True),
            **(
                {"max_grad": spec.seed_policy["max_grad"]}
                if "max_grad" in spec.seed_policy
                else {}
            ),
            **{
                key: spec.seed_policy[key]
                for key in (
                    "loss_timing",
                    "epoch_cursor",
                    "epoch_recurrent_state",
                    "source_curve_reset_each_epoch",
                )
                if key in spec.seed_policy
            },
        },
        "tracking": tracking,
    }
