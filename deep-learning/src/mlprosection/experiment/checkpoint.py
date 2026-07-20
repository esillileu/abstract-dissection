"""Epoch-boundary checkpoints for exact local training resumption."""

from __future__ import annotations

# ruff: noqa: E701

import json
import pickle
import random
from pathlib import Path
from typing import Any

import numpy as np

from mlprosection.nn.layers import Layer


def save_epoch_checkpoint(*, root: Path, model: Layer, optimizer: Any, trainer: Any, config_digest: str) -> Path:
    path = root / f"epoch-{int(trainer.epoch or 0):04d}"
    path.mkdir(parents=True, exist_ok=True)
    model.save_params_npz(path / "model.npz")
    _save_buffers(model, path / "buffers.npz")
    backend_state = _backend_rng_state(model)
    with (path / "state.pkl").open("wb") as file:
        pickle.dump({"optimizer": optimizer.state_dict(), "trainer": trainer.state_dict(), "python_rng": random.getstate(), "numpy_rng": np.random.get_state(), "backend_rng": backend_state}, file, protocol=pickle.HIGHEST_PROTOCOL)
    (path / "manifest.json").write_text(json.dumps({"schema_version": 1, "epoch": trainer.epoch, "global_step": trainer.global_step, "config_digest": config_digest}, indent=2, sort_keys=True), encoding="utf-8")
    return path


def load_epoch_checkpoint(*, path: str | Path, model: Layer, optimizer: Any, trainer: Any, config_digest: str) -> None:
    root = Path(path)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest["config_digest"] != config_digest:
        raise ValueError("checkpoint configuration digest does not match this run")
    model.load_params_npz(root / "model.npz")
    _load_buffers(model, root / "buffers.npz")
    with (root / "state.pkl").open("rb") as file:
        state = pickle.load(file)
    optimizer.load_state_dict(state["optimizer"])
    trainer.load_state_dict(state["trainer"])
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    _restore_backend_rng_state(model, state.get("backend_rng"))


def _save_buffers(model: Layer, path: Path) -> None:
    arrays = {
        name: layer.backend.to_numpy(value).copy()
        for name, layer, _attr, value in _buffer_items(model)
        if value is not None
    }
    np.savez(path, **arrays)


def _load_buffers(model: Layer, path: Path) -> None:
    if not path.exists(): return
    targets = {name: (layer, attr) for name, layer, attr, _ in _buffer_items(model)}
    with np.load(path, allow_pickle=False) as arrays:
        for name in arrays.files:
            if name in targets:
                layer, attr = targets[name]
                setattr(layer, attr, layer.backend.xp.asarray(arrays[name]))


def _buffer_items(root: Layer):
    seen: set[int] = set()
    def walk(value: Any, prefix: str):
        if isinstance(value, Layer):
            if id(value) in seen: return
            seen.add(id(value))
            for attr, child in vars(value).items():
                is_normalization_buffer = attr in {"running_mean", "running_var"}
                is_recurrent_state = (
                    attr in {"h", "c"} and bool(getattr(value, "stateful", False))
                )
                if is_normalization_buffer or is_recurrent_state:
                    yield f"{prefix}/{attr}", value, attr, child
                yield from walk(child, f"{prefix}/{attr}")
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value): yield from walk(child, f"{prefix}/{index}")
        elif isinstance(value, dict):
            for key, child in value.items(): yield from walk(child, f"{prefix}/{key}")
    yield from walk(root, "model")


def _backend_rng_state(model: Layer):
    rng = model.backend.xp.random
    return rng.get_state() if hasattr(rng, "get_state") else None


def _restore_backend_rng_state(model: Layer, state: Any) -> None:
    rng = model.backend.xp.random
    if state is not None and hasattr(rng, "set_state"): rng.set_state(state)
