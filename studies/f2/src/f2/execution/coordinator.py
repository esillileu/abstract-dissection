"""Execution Coordinator: The single lifecycle decision-maker for F2 runs.

Implements ref.md section 1, 3.2 & 4.4:
- Evaluates:
    1. skip: Matching MLflow FINISHED run already exists.
    2. publish: Local complete result exists but not published.
    3. resume: Local checkpoint exists and can be resumed.
    4. new: New attempt created.
- Never performs numerical calculations directly.
- Coordinates the entire execution lifecycle (skip -> local publish -> resume -> run -> publish).
- Delegates storage operations to f2.storage and training to the suite executor.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from f2.execution.identity import ConditionIdentity
from f2.storage.publication import PublicationService
from f2.storage.runs import LocalRunStore
from repro_core.context import ExperimentContext
from repro_core.context.paths import RuntimePaths
from repro_core.storage.local_store import LocalStore


class ExecutionDecisionKind(StrEnum):
    SKIP = "skip"
    PUBLISH = "publish"
    RESUME = "resume"
    RUN = "run"


@dataclass(frozen=True)
class ExecutionDecision:
    kind: ExecutionDecisionKind
    run_dir: Path
    existing_run_id: str | None = None
    resume_checkpoint: Path | None = None
    resume_epoch: int | None = None


@dataclass(frozen=True)
class ExecutionReceipt:
    slot_id: str
    run_dir: Path
    run_id: str | None
    decision: ExecutionDecisionKind
    result: Any | None = None
    skipped: bool = False


class F2ExecutionCoordinator:
    """Sole policy decision-maker for the F2 execution lifecycle."""

    def __init__(
        self,
        tracking_uri: str,
        paths: RuntimePaths | None = None,
    ) -> None:
        self.tracking_uri = tracking_uri
        self.paths = paths or RuntimePaths.from_environment()
        self.publication = PublicationService(tracking_uri=tracking_uri)
        self.run_store = LocalRunStore(paths=self.paths)

    def decide(
        self,
        condition: ConditionIdentity,
        *,
        experiment_name: str,
        override: bool = False,
        atomic_run_id: str | None = None,
    ) -> ExecutionDecision:
        """Determine lifecycle action according to ref.md 4.4."""
        # 1. Skip on existing FINISHED run (unless overridden)
        if not override:
            finished = self.publication.find_matching_finished_run(
                experiment_name,
                condition,
                atomic_run_id=atomic_run_id,
            )
            if finished is not None:
                run_dir = self.run_store.get_run_root(
                    condition.suite_id,
                    condition.planned_run_slot_id,
                    override=False,
                )
                return ExecutionDecision(
                    kind=ExecutionDecisionKind.SKIP,
                    run_dir=run_dir,
                    existing_run_id=finished.info.run_id,
                )

        # 2. Inspect existing attempts for this slot if not override
        if not override:
            attempts = self.run_store.discover_attempts(
                condition.suite_id,
                condition.planned_run_slot_id,
            )
            # Inspect attempts in reverse order (latest attempt first)
            for attempt_dir in reversed(attempts):
                status = self.run_store.inspect_run(attempt_dir)
                if not self.run_store.matches_condition(status, condition):
                    continue

                # Publish unposted local complete result matching exact condition
                if status.is_complete:
                    return ExecutionDecision(
                        kind=ExecutionDecisionKind.PUBLISH,
                        run_dir=attempt_dir,
                    )

                # Resume from checkpoint if valid and matches condition
                if status.is_checkpoint:
                    return ExecutionDecision(
                        kind=ExecutionDecisionKind.RESUME,
                        run_dir=attempt_dir,
                        resume_checkpoint=status.checkpoint_path,
                        resume_epoch=status.epoch,
                    )

        # 3. Fresh run attempt (or override)
        if override:
            run_dir = self.run_store.get_run_root(
                condition.suite_id,
                condition.planned_run_slot_id,
                override=True,
            )
        else:
            # Check if default base directory is already occupied by a different condition or incomplete run
            run_dir = self.run_store.get_isolated_run_root(
                condition.suite_id,
                condition.planned_run_slot_id,
            )
        return ExecutionDecision(
            kind=ExecutionDecisionKind.RUN,
            run_dir=run_dir,
        )

    def execute_condition(
        self,
        condition: ConditionIdentity,
        config: dict[str, Any],
        context: ExperimentContext,
        *,
        experiment_name: str,
        training_delegate: Callable[..., Any],
        override: bool = False,
        progress_reporter: Any | None = None,
        load_result_delegate: Callable[[Path], Any] | None = None,
        decision: ExecutionDecision | None = None,
    ) -> ExecutionReceipt:
        """Own the full execution lifecycle: skip -> publish -> resume -> run -> publish."""
        if decision is None:
            atomic_run_id = str(config.get("atomic_run_id", "")) or None
            decision = self.decide(
                condition,
                experiment_name=experiment_name,
                override=override,
                atomic_run_id=atomic_run_id,
            )

        root_override = context.metadata.get("run_root")
        run_root = Path(root_override) if root_override else decision.run_dir
        local_store = LocalStore(run_root)
        local_store.ensure_directories()

        # Persist canonical config.json BEFORE training begins (ref.md P0-2 & instructions)
        # This guarantees that if training is interrupted, the directory has its condition identity preserved
        if decision.kind in (ExecutionDecisionKind.RUN, ExecutionDecisionKind.RESUME):
            local_store.write_config(config)

        # 1. Skip check: MLflow FINISHED run
        if decision.kind == ExecutionDecisionKind.SKIP and not override:
            existing_run_id = decision.existing_run_id
            if progress_reporter is not None:
                progress_reporter.write(
                    f"Skipping {condition.suite_id} slot={condition.planned_run_slot_id}: "
                    f"finished run {existing_run_id} exists"
                )
            return ExecutionReceipt(
                slot_id=condition.planned_run_slot_id,
                run_dir=run_root,
                run_id=existing_run_id,
                decision=decision.kind,
                result=None,
                skipped=True,
            )

        # 2. Local complete result exists: publish immediately without re-training
        if decision.kind == ExecutionDecisionKind.PUBLISH and not override:
            if progress_reporter is not None:
                progress_reporter.write(
                    f"Publishing existing local result for slot={condition.planned_run_slot_id}"
                )
            run_id = self.publication.publish_run(
                experiment_name,
                condition,
                run_root,
                config,
            )
            result = (
                load_result_delegate(run_root)
                if load_result_delegate is not None
                else self._default_load_result(run_root)
            )
            return ExecutionReceipt(
                slot_id=condition.planned_run_slot_id,
                run_dir=run_root,
                run_id=run_id,
                decision=decision.kind,
                result=result,
                skipped=False,
            )

        # 3. Resume from checkpoint or 4. Fresh run
        resume_from = None
        if decision.kind == ExecutionDecisionKind.RESUME and not override:
            resume_from = decision.resume_checkpoint
            if progress_reporter is not None:
                progress_reporter.write(
                    f"Resuming {condition.suite_id} slot={condition.planned_run_slot_id} "
                    f"from epoch {decision.resume_epoch} checkpoint"
                )

        result = training_delegate(
            config,
            context,
            run_root=run_root,
            local_store=local_store,
            resume_from=resume_from,
            progress_reporter=progress_reporter,
        )

        # 5. Publish to MLflow and finalize
        run_id = self.publication.publish_run(
            experiment_name,
            condition,
            run_root,
            config,
        )
        return ExecutionReceipt(
            slot_id=condition.planned_run_slot_id,
            run_dir=run_root,
            run_id=run_id,
            decision=decision.kind,
            result=result,
            skipped=False,
        )

    def _default_load_result(self, run_root: Path) -> Any:
        res_file = run_root / "result.json"
        if res_file.is_file():
            try:
                return json.loads(res_file.read_text(encoding="utf-8"))
            except Exception:
                return None
        return None


__all__ = [
    "ExecutionDecision",
    "ExecutionDecisionKind",
    "ExecutionReceipt",
    "F2ExecutionCoordinator",
]
