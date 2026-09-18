from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from dlfs.adapters.checkpoint import (
    create_deepscratch_checkpoint_manager,
    load_deepscratch_checkpoint,
)
from repro_core.context.checkpoint import CheckpointManager, CheckpointRetentionPolicy

from ..records import DS1Records


def compute_checkpoint_identity_digest(
    config: dict[str, Any], checkpoint_config: dict[str, Any]
) -> str:
    checkpoint_identity = dict(config)
    checkpoint_identity["checkpoint"] = dict(checkpoint_config)
    checkpoint_identity["checkpoint"].pop("resume", None)
    return hashlib.sha256(
        json.dumps(checkpoint_identity, sort_keys=True, default=str).encode()
    ).hexdigest()


def setup_checkpoint_manager(
    *,
    checkpoint_root: Path,
    checkpoint_config: dict[str, Any],
    config_digest: str,
    model: Any,
    objective: Any,
    optimizer: Any,
    trainer: Any,
) -> CheckpointManager:
    checkpoint_manager = create_deepscratch_checkpoint_manager(
        root=checkpoint_root,
        model=model,
        objective=objective,
        optimizer=optimizer,
        trainer=trainer,
        config_digest=config_digest,
        policy=CheckpointRetentionPolicy.from_mapping(checkpoint_config),
    )
    if resume := checkpoint_config.get("resume"):
        load_deepscratch_checkpoint(
            path=str(resume),
            model=model,
            objective=objective,
            optimizer=optimizer,
            trainer=trainer,
            config_digest=config_digest,
        )
    return checkpoint_manager


def record_manager_checkpoints(
    checkpoint_manager: CheckpointManager, records: DS1Records
) -> None:
    latest = checkpoint_manager.current("latest")
    if latest is not None:
        records.add_checkpoint(
            update=latest.update,
            epoch=latest.epoch,
            kind="latest",
            path=latest.path,
            sha256=latest.sha256,
        )
    best = checkpoint_manager.current("best")
    if best is not None:
        records.add_checkpoint(
            update=best.update,
            epoch=best.epoch,
            kind="selected",
            path=best.path,
            sha256=best.sha256,
        )
    for periodic in checkpoint_manager.retained_periodic():
        records.add_checkpoint(
            update=periodic.update,
            epoch=periodic.epoch,
            kind="periodic",
            path=periodic.path,
            sha256=periodic.sha256,
        )
    records.flush()
