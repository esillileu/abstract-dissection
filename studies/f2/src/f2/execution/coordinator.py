"""Execution Coordinator: The single lifecycle decision-maker for F2 runs.

Implements ref.md section 1, 3.2 & 4.4:
- Evaluates:
    1. skip: Matching MLflow FINISHED run already exists.
    2. publish: Local complete result exists but not published.
    3. resume: Local checkpoint exists and can be resumed.
    4. new: New attempt created.
- Never performs numerical calculations directly.
- Delegates storage operations to f2.storage and training to the suite executor.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from f2.execution.identity import ConditionIdentity
from f2.storage.publication import PublicationService
from f2.storage.runs import LocalRunStore
from repro_core.context.paths import RuntimePaths


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
        run_dir = self.run_store.get_run_root(
            condition.suite_id,
            condition.planned_run_slot_id,
            override=override,
        )

        # 1. Skip on existing FINISHED run (unless overridden)
        if not override:
            finished = self.publication.find_matching_finished_run(
                experiment_name,
                condition,
                atomic_run_id=atomic_run_id,
            )
            if finished is not None:
                return ExecutionDecision(
                    kind=ExecutionDecisionKind.SKIP,
                    run_dir=run_dir,
                    existing_run_id=finished.info.run_id,
                )

        # 2. Inspect local attempt state
        local_status = self.run_store.inspect_run(run_dir)

        # 3. Publish unposted local complete result (if not override)
        if local_status.is_complete and not override:
            return ExecutionDecision(
                kind=ExecutionDecisionKind.PUBLISH,
                run_dir=run_dir,
            )

        # 4. Resume from checkpoint if valid and not override
        if local_status.is_checkpoint and not override:
            return ExecutionDecision(
                kind=ExecutionDecisionKind.RESUME,
                run_dir=run_dir,
                resume_checkpoint=local_status.checkpoint_path,
                resume_epoch=local_status.epoch,
            )

        # 5. Fresh run attempt
        return ExecutionDecision(
            kind=ExecutionDecisionKind.RUN,
            run_dir=run_dir,
        )


__all__ = [
    "ExecutionDecision",
    "ExecutionDecisionKind",
    "ExecutionReceipt",
    "F2ExecutionCoordinator",
]
