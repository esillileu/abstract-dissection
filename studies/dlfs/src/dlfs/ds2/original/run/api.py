"""Cache-aware DS2 original trial execution."""

from __future__ import annotations

import importlib
import shutil
from pathlib import Path

from dlfs.original_runtime.cache_protocol import (
    cache_is_valid,
    publish_result,
    stable_hash,
    staging_directory,
)
from dlfs.original_runtime.worktree import (
    clean_worktree,
    head_commit,
    source_hashes,
)
from repro_core.context.paths import RuntimePaths

from .common import Trial

UPSTREAM = (
    RuntimePaths.from_environment().reference("dlfs2-book") / "source"
).resolve()
RUNNER_SCHEMA_VERSION = 2


def trials_for(experiments: list[str]) -> list[tuple[str, Trial]]:
    trials = []
    for experiment in experiments:
        module = importlib.import_module(f"dlfs.ds2.original.run.{experiment}")
        trials.extend((experiment, trial) for trial in module.TRIALS)
    return trials


def run(experiments: list[str], *, root: Path, force: bool = False) -> None:
    commit = head_commit(UPSTREAM)
    dirty = not clean_worktree(UPSTREAM)
    sources = source_hashes(UPSTREAM)
    for experiment, trial in trials_for(experiments):
        target = root / "data" / experiment / trial.trial_id
        config_hash = stable_hash(
            {
                "schema_version": RUNNER_SCHEMA_VERSION,
                "trial_id": trial.trial_id,
                "seed": trial.seed,
                "conditions": trial.conditions,
                "dirty_worktree": dirty,
            }
        )
        if not force and cache_is_valid(target, {"config_hash": config_hash}):
            continue

        staging = staging_directory(target)
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True, exist_ok=True)
        trial.executor(staging)
        publish_result(
            staging,
            target,
            identity={
                "seed": trial.seed,
                "backend": trial.backend,
                "upstream_commit": commit,
                "source_hashes": sources,
                "conditions": trial.conditions,
                "config_hash": config_hash,
            },
        )
