"""Backward-compatible re-export of training event contracts."""

from deepscratch.trainer.events import NullTrainerCallback, TrainerCallback

__all__ = ["NullTrainerCallback", "TrainerCallback"]
