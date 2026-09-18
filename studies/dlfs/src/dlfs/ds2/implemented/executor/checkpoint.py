from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from repro_core.context import ExperimentContext
from repro_core.context.checkpoint import CheckpointManager

from ..records import DS2Records
from .common import _mapping


def _publish_array_checkpoint(context: ExperimentContext, **arrays: np.ndarray) -> Path:
    """Publish analysis arrays through the canonical v2 checkpoint pointer."""
    root = Path(str(context.metadata["checkpoint_root"]))
    root.mkdir(parents=True, exist_ok=True)
    target = root / "final.npz"
    temporary = root / ".final.npz.tmp"
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, target)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    pointer = {
        "schema_version": 2,
        "role": "latest",
        "path": target.name,
        "sha256": digest,
        "epoch": 0,
        "update": 0,
    }
    temporary_pointer = root / ".latest.json.tmp"
    temporary_pointer.write_text(
        json.dumps(pointer, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary_pointer, root / "latest.json")
    return target


def _config_digest(config: dict[str, object]) -> str:
    checkpoint_config = dict(_mapping(config, "checkpoint"))
    checkpoint_config.pop("resume", None)
    identity = dict(config)
    identity["checkpoint"] = checkpoint_config
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, default=str).encode()
    ).hexdigest()


def _save_epoch_roles(manager: CheckpointManager) -> None:
    if manager.policy.save_latest:
        manager.save_latest()
    manager.save_periodic_if_due()


def _record_retained_checkpoints(
    records: DS2Records,
    manager: CheckpointManager,
    *,
    best_metric: str = "",
    best_value: float | None = None,
) -> None:
    final = manager.current("final")
    if final is not None:
        records.add_checkpoint(
            update=final.update,
            epoch=final.epoch,
            kind="final",
            path=final.path,
            sha256=final.sha256,
            checkpoint_id="final",
        )
    latest = manager.current("latest")
    if latest is not None:
        records.add_checkpoint(
            update=latest.update,
            epoch=latest.epoch,
            kind="latest",
            path=latest.path,
            sha256=latest.sha256,
            checkpoint_id=f"latest-epoch-{latest.epoch:04d}",
        )
    best = manager.current("best")
    if best is not None:
        records.add_checkpoint(
            update=best.update,
            epoch=best.epoch,
            kind="selected",
            path=best.path,
            sha256=best.sha256,
            checkpoint_id=f"selected-epoch-{best.epoch:04d}",
            selection_metric=best_metric,
            selection_value="" if best_value is None else best_value,
        )
    for periodic in manager.retained_periodic():
        records.add_checkpoint(
            update=periodic.update,
            epoch=periodic.epoch,
            kind="periodic",
            path=periodic.path,
            sha256=periodic.sha256,
            checkpoint_id=f"periodic-epoch-{periodic.epoch:04d}",
        )
