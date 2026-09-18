from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np

from repro_core.context.checkpoint import (
    CheckpointManager,
    CheckpointRetentionPolicy,
    resolve_checkpoint_path,
)

from .state import (
    _backend_rng_state,
    _load_buffers,
    _read_pickle,
    _restore_backend_rng_state,
    _save_buffers,
    _write_pickle,
)


def write_deepscratch_checkpoint(
    path: Path,
    *,
    model: Any,
    objective: Any,
    optimizer: Any,
    trainer: Any,
    config_digest: str,
    payload: str = "full",
) -> None:
    """Serialize DeepScratch model, objective, optimizer, trainer, and RNG state."""
    path.mkdir(parents=True, exist_ok=True)
    model.save_params_npz(path / "model_parameters.npz")
    if payload == "model_only":
        (path / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "payload": payload,
                    "epoch": trainer.epoch,
                    "global_step": trainer.global_step,
                    "config_digest": config_digest,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return
    _save_buffers(model, path / "model_buffers.npz")
    objective.save_params_npz(path / "objective_parameters.npz")
    _save_buffers(objective, path / "objective_buffers.npz")
    backend_state = _backend_rng_state(model)
    _write_pickle(path / "optimizer_state.pkl", optimizer.state_dict())
    _write_pickle(path / "trainer_state.pkl", trainer.state_dict())
    _write_pickle(
        path / "rng_state.pkl",
        {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "backend": backend_state,
        },
    )
    (path / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "payload": payload,
                "epoch": trainer.epoch,
                "global_step": trainer.global_step,
                "config_digest": config_digest,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def load_deepscratch_checkpoint(
    *,
    path: str | Path,
    model: Any,
    objective: Any,
    optimizer: Any,
    trainer: Any,
    config_digest: str,
) -> None:
    """Restore DeepScratch training state from a v2 epoch checkpoint."""
    root = resolve_checkpoint_path(Path(path))
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 2:
        raise ValueError("only checkpoint schema version 2 is supported")
    if manifest["config_digest"] != config_digest:
        raise ValueError("checkpoint configuration digest does not match this run")
    model.load_params_npz(root / "model_parameters.npz")
    _load_buffers(model, root / "model_buffers.npz")
    objective.load_params_npz(root / "objective_parameters.npz")
    _load_buffers(objective, root / "objective_buffers.npz")
    optimizer.load_state_dict(_read_pickle(root / "optimizer_state.pkl"))
    trainer.load_state_dict(_read_pickle(root / "trainer_state.pkl"))
    rng_state = _read_pickle(root / "rng_state.pkl")
    random.setstate(rng_state["python"])
    np.random.set_state(rng_state["numpy"])
    _restore_backend_rng_state(model, rng_state.get("backend"))


def save_deepscratch_epoch_checkpoint(
    *,
    root: Path,
    model: Any,
    objective: Any,
    optimizer: Any,
    trainer: Any,
    config_digest: str,
) -> Path:
    """Write an unmanaged epoch checkpoint directly to the specified root directory."""
    path = root / f"epoch-{int(trainer.epoch or 0):04d}"
    write_deepscratch_checkpoint(
        path=path,
        model=model,
        objective=objective,
        optimizer=optimizer,
        trainer=trainer,
        config_digest=config_digest,
    )
    return path


def create_deepscratch_checkpoint_manager(
    root: Path,
    *,
    model: Any,
    objective: Any,
    optimizer: Any,
    trainer: Any,
    config_digest: str,
    policy: CheckpointRetentionPolicy | None = None,
) -> CheckpointManager:
    """Bind DeepScratch serialization to CheckpointManager generational retention."""

    def save_fn(path: Path, payload: str) -> None:
        write_deepscratch_checkpoint(
            path,
            model=model,
            objective=objective,
            optimizer=optimizer,
            trainer=trainer,
            config_digest=config_digest,
            payload=payload,
        )

    return CheckpointManager(
        root=root,
        config_digest=config_digest,
        save_fn=save_fn,
        epoch_fn=lambda: int(trainer.epoch or 0),
        step_fn=lambda: int(trainer.global_step or 0),
        policy=policy,
    )
