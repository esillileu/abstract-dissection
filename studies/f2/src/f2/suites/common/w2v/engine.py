"""Pure computation engine for Word2Vec training execution."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EngineExecutionResult:
    """Computation outputs and progress reports produced by the compute engine."""

    completed_epochs: int
    is_complete: bool
    reports: list[dict[str, Any]]
    session: Any


class Word2VecEngine:
    """Pure compute layer for Word2Vec model training.

    Responsible only for model computation, epoch training loops, and yielding
    progress/observation events via callbacks.
    Does not determine storage paths, inspect local storage, access MLflow,
    or manage caches.
    """

    def train(
        self,
        session: Any,
        *,
        total_epochs: int,
        on_epoch_complete: Callable[[int, Any, list[dict[str, Any]]], None]
        | None = None,
        on_observation: Callable[[list[dict[str, Any]]], None] | None = None,
        progress_callback: Callable[[str], None] | None = None,
        existing_reports: list[dict[str, Any]] | None = None,
    ) -> EngineExecutionResult:
        reports: list[dict[str, Any]] = (
            list(existing_reports) if existing_reports else []
        )

        while not session.is_complete:
            epoch = session.train_epoch()

            obs = epoch.observations()
            if on_observation is not None:
                on_observation(obs)

            epoch_record = {
                "epoch": epoch.epoch,
                "processed_tokens": epoch.processed_tokens,
                "objective_loss": epoch.objective_loss,
            }
            reports.append(epoch_record)

            if on_epoch_complete is not None:
                on_epoch_complete(epoch.epoch, epoch, reports)

            if progress_callback is not None:
                progress_callback(
                    f"epoch={epoch.epoch}/{total_epochs} "
                    f"tokens={epoch.processed_tokens} "
                    f"loss={epoch.objective_loss:.4f} "
                    f"tokens_per_second={epoch.tokens_per_second:.1f}"
                )

        return EngineExecutionResult(
            completed_epochs=session.completed_epochs,
            is_complete=session.is_complete,
            reports=reports,
            session=session,
        )


__all__ = ["EngineExecutionResult", "Word2VecEngine"]
