"""DeepScratch state serialization and deserialization adapter for checkpoints."""

from __future__ import annotations

import json
import pickle
import random
import re
from pathlib import Path
from typing import Any

import numpy as np

from repro_core.context.checkpoint import (
    CheckpointManager,
    CheckpointRetentionPolicy,
    resolve_checkpoint_path,
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


def load_deepscratch_model_parameters(path: str | Path, model: Any) -> Path:
    """Load only model state from a v2 checkpoint or legacy parameter archive."""
    resolved = resolve_checkpoint_path(path)
    if resolved.is_file() and resolved.suffix == ".npz":
        _load_model_parameters_compatible(model, resolved)
        return resolved
    if not resolved.is_dir():
        raise ValueError(f"model checkpoint does not exist: {resolved}")
    parameters = resolved / "model_parameters.npz"
    if not parameters.is_file():
        raise ValueError(f"model checkpoint parameters are missing: {parameters}")
    _load_model_parameters_compatible(model, parameters)
    _load_buffers(model, resolved / "model_buffers.npz")
    return resolved


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


def _load_model_parameters_compatible(model: Any, path: Path) -> None:
    """Strictly load parameters while ignoring legacy forward-cache entries."""
    current = {name for name, _parameter in model.named_parameters()}
    with np.load(path, allow_pickle=False) as archive:
        saved = set(archive.files)
    missing = current - saved
    unexpected = saved - current
    unsupported = {
        name for name in unexpected if re.fullmatch(r"layers\.\d+\.x", name) is None
    }
    if missing:
        raise KeyError(f"missing parameters: {sorted(missing)}")
    if unsupported:
        raise KeyError(f"unexpected parameters: {sorted(unsupported)}")
    model.load_params_npz(path, strict=False)


def _save_buffers(model: Any, path: Path) -> None:
    arrays = {
        name: model.backend.to_numpy(value).copy()
        for name, value in model.named_buffers()
        if value is not None
    }
    np.savez(path, **arrays)


def _resolve_owner(root: Any, name: str) -> tuple[Any, str]:
    parts = name.split(".")
    target = root
    for part in parts[:-1]:
        if isinstance(target, (list, tuple)) or (
            hasattr(target, "__getitem__") and part.isdigit()
        ):
            target = target[int(part)]
        else:
            target = getattr(target, part)
    return target, parts[-1]


def _load_buffers(model: Any, path: Path) -> None:
    if not path.exists():
        return

    targets = {
        name: _resolve_owner(model, name) for name, _value in model.named_buffers()
    }
    with np.load(path, allow_pickle=False) as arrays:
        for name in arrays.files:
            if name in targets:
                layer, attr = targets[name]
                setattr(layer, attr, layer.backend.xp.asarray(arrays[name]))


def _backend_rng_state(model: Any) -> dict[str, Any]:
    rng = model.backend.xp.random
    return {
        "global": rng.get_state() if hasattr(rng, "get_state") else None,
        "streams": model.backend.random_stream_states(),
    }


def _restore_backend_rng_state(model: Any, state: Any) -> None:
    rng = model.backend.xp.random
    if isinstance(state, dict) and "streams" in state:
        global_state = state.get("global")
        if global_state is not None and hasattr(rng, "set_state"):
            rng.set_state(global_state)
        model.backend.restore_random_stream_states(state["streams"])
    elif state is not None and hasattr(rng, "set_state"):
        rng.set_state(state)


def _write_pickle(path: Path, value: Any) -> None:
    with path.open("wb") as file:
        pickle.dump(value, file, protocol=pickle.HIGHEST_PROTOCOL)


def _read_pickle(path: Path) -> Any:
    with path.open("rb") as file:
        return pickle.load(file)
