"""Epoch-boundary checkpoint retention policy and generational pointer management."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CheckpointRetentionPolicy:
    """Global retention contract for resumable training checkpoints.

    ``best`` and ``latest`` are fixed single-generation roles.  Periodic
    generations only exist when both cadence and retention are explicitly
    configured.
    """

    periodic_every_epochs: int | None = None
    periodic_keep: int = 0
    save_latest: bool = True
    best_payload: str = "full"

    def __post_init__(self) -> None:
        if self.periodic_every_epochs is not None and self.periodic_every_epochs < 1:
            raise ValueError("periodic_every_epochs must be positive")
        if self.periodic_keep < 0:
            raise ValueError("periodic_keep must be non-negative")
        if (self.periodic_every_epochs is None) != (self.periodic_keep == 0):
            raise ValueError(
                "periodic checkpoints require both periodic_every_epochs and periodic_keep"
            )
        if self.best_payload not in {"full", "model_only"}:
            raise ValueError("best_payload must be full or model_only")

    @classmethod
    def from_mapping(
        cls, values: dict[str, object] | None
    ) -> CheckpointRetentionPolicy:
        values = values or {}
        retention = values.get("retention", {})
        if not isinstance(retention, dict):
            raise ValueError("checkpoint.retention must be a mapping")
        every = retention.get("periodic_every_epochs")
        keep = retention.get("periodic_keep", 0)
        return cls(
            periodic_every_epochs=None if every is None else int(every),
            periodic_keep=int(keep),
            save_latest=bool(values.get("save_latest", True)),
            best_payload=str(values.get("best_payload", "full")),
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
        config_digest: str,
        save_fn: Callable[[Path, str], None],
        epoch_fn: Callable[[], int],
        step_fn: Callable[[], int],
        policy: CheckpointRetentionPolicy | None = None,
    ) -> None:
        self.root = Path(root)
        self.config_digest = config_digest
        self._save_fn = save_fn
        self._epoch_fn = epoch_fn
        self._step_fn = step_fn
        self.policy = policy or CheckpointRetentionPolicy()
        self._refs: dict[str, CheckpointRef] = {}

    def save_latest(self) -> CheckpointRef:
        return self._save_role("latest", keep=1)

    def save_best(self) -> CheckpointRef:
        return self._save_role("best", keep=1, payload=self.policy.best_payload)

    def save_final(self) -> CheckpointRef:
        """Save the terminal state as the only full resumable checkpoint."""
        return self._save_role("final", keep=1, payload="full")

    def save_periodic_if_due(self) -> CheckpointRef | None:
        every = self.policy.periodic_every_epochs
        if every is None or (self._epoch_fn() % every):
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
            refs.append(
                CheckpointRef(
                    role="periodic",
                    path=path,
                    sha256=_path_digest(path),
                    epoch=int(manifest["epoch"]),
                    update=int(manifest["global_step"]),
                )
            )
        return tuple(refs)

    def _save_role(
        self, role: str, *, keep: int, payload: str = "full"
    ) -> CheckpointRef:
        self.root.mkdir(parents=True, exist_ok=True)
        generations = self.root / "generations"
        generations.mkdir(exist_ok=True)
        suffix = uuid.uuid4().hex[:12]
        epoch = self._epoch_fn()
        step = self._step_fn()
        name = f"{role}-epoch-{int(epoch):04d}-update-{int(step):08d}-{suffix}"
        staging = generations / f".{name}.tmp"
        target = generations / name
        try:
            self._save_fn(staging, payload)
            staging.replace(target)
            ref = CheckpointRef(
                role=role,
                path=target,
                sha256=_path_digest(target),
                epoch=int(epoch),
                update=int(step),
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
        temporary.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "role": ref.role,
                    "path": ref.path.relative_to(self.root).as_posix(),
                    "sha256": ref.sha256,
                    "epoch": ref.epoch,
                    "update": ref.update,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
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


def resolve_checkpoint_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_file() and path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return path.parent / str(payload["path"])
    return path


def _path_digest(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(value for value in path.rglob("*") if value.is_file()):
        digest.update(item.relative_to(path).as_posix().encode())
        with item.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "CheckpointManager",
    "CheckpointRef",
    "CheckpointRetentionPolicy",
    "resolve_checkpoint_path",
]
