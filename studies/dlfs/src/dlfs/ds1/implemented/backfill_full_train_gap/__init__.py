"""Backfill DS1 full-train accuracy and train/test gap metrics in MLflow.

The command is a dry-run unless ``--apply`` is supplied. It never starts new
experiments; it evaluates checkpoints belonging to existing completed runs.
"""

from __future__ import annotations

from .backfill import (
    TARGET_NAMES,
    backfill_runs,
    latest_target_runs,
)
from .cli import main
from .evaluation import (
    _download_file,
    _mapping,
    _run_config,
    evaluate_run,
    resolve_run_checkpoint,
)

__all__ = [
    "TARGET_NAMES",
    "_download_file",
    "_mapping",
    "_run_config",
    "backfill_runs",
    "evaluate_run",
    "latest_target_runs",
    "main",
    "resolve_run_checkpoint",
]
