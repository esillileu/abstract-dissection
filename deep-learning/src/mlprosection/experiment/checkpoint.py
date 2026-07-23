"""Epoch-boundary checkpoints for exact local training resumption."""

from __future__ import annotations

# ruff: noqa: E701

import json
import os
import pickle
import random
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mlprosection.nn.layers import Layer


@dataclass(frozen=True)
class CheckpointRetentionPolicy:
    """Global retention contract for resumable training checkpoints.

    ``best`` and ``latest`` are fixed single-generation roles.  Periodic
    generations only exist when both cadence and retention are explicitly
    configured.
    """

    periodic_every_epochs: int | None = None
    periodic_keep: int = 0

    def __post_init__(self) -> None:
        if self.periodic_every_epochs is not None and self.periodic_every_epochs < 1:
            raise ValueError("periodic_every_epochs must be positive")
        if self.periodic_keep < 0:
            raise ValueError("periodic_keep must be non-negative")
        if (self.periodic_every_epochs is None) != (self.periodic_keep == 0):
            raise ValueError(
                "periodic checkpoints require both periodic_every_epochs and periodic_keep"
            )

    @classmethod
    def from_mapping(cls, values: dict[str, object] | None) -> CheckpointRetentionPolicy:
        values = values or {}
        retention = values.get("retention", {})
        if not isinstance(retention, dict):
            raise ValueError("checkpoint.retention must be a mapping")
        every = retention.get("periodic_every_epochs")
        keep = retention.get("periodic_keep", 0)
        return cls(
            periodic_every_epochs=None if every is None else int(every),
            periodic_keep=int(keep),
        )


@dataclass(frozen=True)
class CheckpointRef:
    role: str
    path: Path
    sha256: str
    epoch: int
    update: int


class CheckpointManager:
    """Atomically publish and retain checkpoint generations by semantic role."""

    def __init__(
        self,
        *,
        root: Path,
        model: Layer,
        objective: Layer,
        optimizer: Any,
        trainer: Any,
        config_digest: str,
        policy: CheckpointRetentionPolicy | None = None,
    ) -> None:
        self.root = Path(root)
        self.model = model
        self.objective = objective
        self.optimizer = optimizer
        self.trainer = trainer
        self.config_digest = config_digest
        self.policy = policy or CheckpointRetentionPolicy()
        self._refs: dict[str, CheckpointRef] = {}

    def save_latest(self) -> CheckpointRef:
        return self._save_role("latest", keep=1)

    def save_best(self) -> CheckpointRef:
        return self._save_role("best", keep=1)

    def save_periodic_if_due(self) -> CheckpointRef | None:
        every = self.policy.periodic_every_epochs
        if every is None or self.trainer.epoch % every:
            return None
        return self._save_role("periodic", keep=self.policy.periodic_keep)

    def current(self, role: str) -> CheckpointRef | None:
        if role in self._refs:
            return self._refs[role]
        pointer = self.root / f"{role}.json"
        if not pointer.exists():
            return None
        payload = json.loads(pointer.read_text(encoding="utf-8"))
        ref = CheckpointRef(
            role=role,
            path=self.root / str(payload["path"]),
            sha256=str(payload["sha256"]),
            epoch=int(payload["epoch"]),
            update=int(payload["update"]),
        )
        self._refs[role] = ref
        return ref

    def retained_periodic(self) -> tuple[CheckpointRef, ...]:
        refs = []
        for path in self._generation_paths("periodic"):
            manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
            refs.append(CheckpointRef(
                role="periodic", path=path, sha256=_path_digest(path),
                epoch=int(manifest["epoch"]), update=int(manifest["global_step"]),
            ))
        return tuple(refs)

    def _save_role(self, role: str, *, keep: int) -> CheckpointRef:
        self.root.mkdir(parents=True, exist_ok=True)
        generations = self.root / "generations"
        generations.mkdir(exist_ok=True)
        suffix = uuid.uuid4().hex[:12]
        name = (
            f"{role}-epoch-{int(self.trainer.epoch):04d}"
            f"-update-{int(self.trainer.global_step):08d}-{suffix}"
        )
        staging = generations / f".{name}.tmp"
        target = generations / name
        try:
            _write_epoch_checkpoint(
                path=staging,
                model=self.model,
                objective=self.objective,
                optimizer=self.optimizer,
                trainer=self.trainer,
                config_digest=self.config_digest,
            )
            staging.replace(target)
            ref = CheckpointRef(
                role=role,
                path=target,
                sha256=_path_digest(target),
                epoch=int(self.trainer.epoch),
                update=int(self.trainer.global_step),
            )
            self._publish_pointer(ref)
            self._refs[role] = ref
            self._prune(role, keep=keep, preserve=target)
            return ref
        except BaseException:
            if staging.exists():
                shutil.rmtree(staging)
            raise

    def _publish_pointer(self, ref: CheckpointRef) -> None:
        pointer = self.root / f"{ref.role}.json"
        temporary = self.root / f".{ref.role}.{uuid.uuid4().hex}.tmp"
        temporary.write_text(json.dumps({
            "schema_version": 2,
            "role": ref.role,
            "path": ref.path.relative_to(self.root).as_posix(),
            "sha256": ref.sha256,
            "epoch": ref.epoch,
            "update": ref.update,
        }, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, pointer)

    def _generation_paths(self, role: str) -> list[Path]:
        generations = self.root / "generations"
        if not generations.exists():
            return []
        return sorted(
            (path for path in generations.glob(f"{role}-*") if path.is_dir()),
            key=lambda path: path.name,
        )

    def _prune(self, role: str, *, keep: int, preserve: Path) -> None:
        paths = self._generation_paths(role)
        retained = set(paths[-keep:]) if keep else set()
        retained.add(preserve)
        for path in paths:
            if path not in retained:
                shutil.rmtree(path)


def save_epoch_checkpoint(*, root: Path, model: Layer, objective: Layer, optimizer: Any, trainer: Any, config_digest: str) -> Path:
    path = root / f"epoch-{int(trainer.epoch or 0):04d}"
    _write_epoch_checkpoint(
        path=path, model=model, objective=objective, optimizer=optimizer, trainer=trainer,
        config_digest=config_digest,
    )
    return path


def _write_epoch_checkpoint(*, path: Path, model: Layer, objective: Layer, optimizer: Any, trainer: Any, config_digest: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    model.save_params_npz(path / "model_parameters.npz")
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
    (path / "manifest.json").write_text(json.dumps({"schema_version": 2, "epoch": trainer.epoch, "global_step": trainer.global_step, "config_digest": config_digest}, indent=2, sort_keys=True), encoding="utf-8")


def load_epoch_checkpoint(*, path: str | Path, model: Layer, objective: Layer, optimizer: Any, trainer: Any, config_digest: str) -> None:
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


def resolve_checkpoint_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_file() and path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return path.parent / str(payload["path"])
    return path


def _write_pickle(path: Path, value: Any) -> None:
    with path.open("wb") as file:
        pickle.dump(value, file, protocol=pickle.HIGHEST_PROTOCOL)


def _read_pickle(path: Path) -> Any:
    with path.open("rb") as file:
        return pickle.load(file)


def _path_digest(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    for item in sorted(value for value in path.rglob("*") if value.is_file()):
        digest.update(item.relative_to(path).as_posix().encode())
        with item.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _save_buffers(model: Layer, path: Path) -> None:
    arrays = {
        name: model.backend.to_numpy(value).copy()
        for name, value in model.named_buffers()
        if value is not None
    }
    np.savez(path, **arrays)


def _load_buffers(model: Layer, path: Path) -> None:
    if not path.exists(): return
    from mlprosection.nn.layers.base import _resolve_owner

    targets = {
        name: _resolve_owner(model, name)
        for name, _value in model.named_buffers()
    }
    with np.load(path, allow_pickle=False) as arrays:
        for name in arrays.files:
            if name in targets:
                layer, attr = targets[name]
                setattr(layer, attr, layer.backend.xp.asarray(arrays[name]))


def _backend_rng_state(model: Layer):
    rng = model.backend.xp.random
    return rng.get_state() if hasattr(rng, "get_state") else None


def _restore_backend_rng_state(model: Layer, state: Any) -> None:
    rng = model.backend.xp.random
    if state is not None and hasattr(rng, "set_state"): rng.set_state(state)
