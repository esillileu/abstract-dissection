"""Cache-aware DS2 original trial execution."""

from __future__ import annotations

import importlib
import shutil
from pathlib import Path

from exp.deepscratch.original_runtime.cache_protocol import cache_is_valid, publish_result, stable_hash, staging_directory
from exp.deepscratch.original_runtime.worktree import clean_worktree, head_commit, source_hashes

from .common import Trial


UPSTREAM = Path(
    "01_deep-learning-from-base/deep-learning-from-scratch-2"
).resolve()
RUNNER_SCHEMA_VERSION = 2


def trials_for(experiments: list[str]) -> list[tuple[str, Trial]]:
    trials = []
    for experiment in experiments:
        module = importlib.import_module(f"exp.deepscratch.ds2.original.run.{experiment}")
        trials.extend((experiment, trial) for trial in module.TRIALS)
    return trials


def run(experiments: list[str], *, root: Path, force: bool = False) -> None:
    commit = head_commit(UPSTREAM)
    pending = []
    for experiment, trial in trials_for(experiments):
        identity = _identity(trial, commit)
        target = root / "data" / experiment / trial.trial_id
        if not force and cache_is_valid(target, identity):
            print(f"{experiment}/{trial.trial_id}: cache hit", flush=True)
            continue
        pending.append((experiment, trial, identity, target))
    if not pending:
        return
    with clean_worktree(UPSTREAM) as worktree:
        for experiment, trial, identity, target in pending:
            print(f"{experiment}/{trial.trial_id}: running ({trial.backend})", flush=True)
            staging = staging_directory(target)
            try:
                trial.runner(worktree, staging, root)
                publish_result(staging, target, identity=identity)
            except BaseException:
                shutil.rmtree(staging, ignore_errors=True)
                raise


def _identity(trial: Trial, commit: str) -> dict[str, object]:
    hashes = source_hashes(UPSTREAM, trial.source_files)
    return {
        "runner_schema_version": RUNNER_SCHEMA_VERSION,
        "seed": 1,
        "backend": trial.backend,
        "upstream_commit": commit,
        "source_hashes": hashes,
        "conditions": trial.conditions,
        "config_hash": stable_hash(
            {
                "runner_schema_version": RUNNER_SCHEMA_VERSION,
                "seed": 1,
                "backend": trial.backend,
                "conditions": trial.conditions,
            }
        ),
    }
