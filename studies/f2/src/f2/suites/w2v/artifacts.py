"""Validated W2V checkpoint and read-only lookup artifact adapters."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from w2v import TrainingState, VocabularyState, WorkerState

from repro_core.context.checkpoint import (
    CheckpointManager,
    CheckpointRetentionPolicy,
)

CHECKPOINT_FORMAT = "f2-w2v-checkpoint-v1"
LOOKUP_FORMAT = "f2-w2v-lookup-v1"

_CHECKPOINT_ARRAYS = (
    "input_embeddings.npy",
    "output_embeddings.npy",
    "token_bytes.npy",
    "token_offsets.npy",
    "counts.npy",
    "huffman_offsets.npy",
    "huffman_paths.npy",
    "huffman_bits.npy",
)
_LOOKUP_ARRAYS = (
    "input_embeddings.npy",
    "token_bytes.npy",
    "token_offsets.npy",
    "counts.npy",
)


def save_checkpoint(
    state: TrainingState,
    path: Path,
    *,
    resource_version: str,
    payload: str = "full",
) -> None:
    """Write one complete checkpoint generation into manager-owned staging."""
    if payload != "full":
        raise ValueError("W2V checkpoints only support the full resumable payload")
    path.mkdir(parents=True, exist_ok=False)
    vocabulary = state.vocabulary_state()
    arrays = vocabulary.arrays()
    values = {
        "input_embeddings.npy": state.input_embeddings(),
        "output_embeddings.npy": state.output_embeddings(),
        "token_bytes.npy": arrays[0],
        "token_offsets.npy": arrays[1],
        "counts.npy": arrays[2],
        "huffman_offsets.npy": arrays[3],
        "huffman_paths.npy": arrays[4],
        "huffman_bits.npy": arrays[5],
    }
    for name, value in values.items():
        np.save(path / name, value, allow_pickle=False)

    trainer = {
        "schema_version": state.schema_version,
        "config_digest": state.config_digest,
        "vocabulary_digest": state.vocabulary_digest,
        "corpus_digest": state.corpus_digest,
        "resource_version": resource_version,
        "completed_epochs": state.completed_epochs,
        "processed_tokens": state.processed_tokens,
        "vocabulary": {
            "hash_capacity": vocabulary.hash_capacity,
            "retained_token_count": vocabulary.retained_token_count,
        },
        "workers": [_worker_record(worker) for worker in state.workers()],
    }
    _write_json(path / "trainer_state.json", trainer)
    _write_manifest(path, CHECKPOINT_FORMAT, trainer, _CHECKPOINT_ARRAYS)


def load_checkpoint(
    path: Path,
    *,
    expected: dict[str, str] | None = None,
) -> TrainingState:
    """Verify every byte and identity before constructing an engine state."""
    manifest = verify_artifact(path, expected=expected, format_name=CHECKPOINT_FORMAT)
    trainer = _read_json(path / "trainer_state.json")
    _validate_manifest_identity(manifest, trainer)
    arrays = {name: _load_array(path / name) for name in _CHECKPOINT_ARRAYS}
    vocabulary = VocabularyState.from_arrays(
        arrays["token_bytes.npy"],
        arrays["token_offsets.npy"],
        arrays["counts.npy"],
        arrays["huffman_offsets.npy"],
        arrays["huffman_paths.npy"],
        arrays["huffman_bits.npy"],
        int(trainer["vocabulary"]["hash_capacity"]),
        int(trainer["vocabulary"]["retained_token_count"]),
    )
    workers = [WorkerState(**record) for record in trainer["workers"]]
    return TrainingState.from_parts(
        int(trainer["schema_version"]),
        str(trainer["config_digest"]),
        str(trainer["vocabulary_digest"]),
        str(trainer["corpus_digest"]),
        int(trainer["completed_epochs"]),
        int(trainer["processed_tokens"]),
        vocabulary,
        arrays["input_embeddings.npy"],
        arrays["output_embeddings.npy"],
        workers,
    )


def save_lookup_artifact(
    state: TrainingState, path: Path, *, resource_version: str
) -> None:
    """Write the compact consumer artifact, excluding resumable trainer state."""
    path.mkdir(parents=True, exist_ok=False)
    vocabulary = state.vocabulary_state()
    token_bytes, token_offsets, counts, *_ = vocabulary.arrays()
    values = {
        "input_embeddings.npy": state.input_embeddings(),
        "token_bytes.npy": token_bytes,
        "token_offsets.npy": token_offsets,
        "counts.npy": counts,
    }
    for name, value in values.items():
        np.save(path / name, value, allow_pickle=False)
    identity = {
        "schema_version": state.schema_version,
        "config_digest": state.config_digest,
        "vocabulary_digest": state.vocabulary_digest,
        "corpus_digest": state.corpus_digest,
        "resource_version": resource_version,
        "completed_epochs": state.completed_epochs,
        "processed_tokens": state.processed_tokens,
    }
    _write_manifest(path, LOOKUP_FORMAT, identity, _LOOKUP_ARRAYS)


@dataclass
class LookupEmbeddings:
    """Memory-mapped input embeddings indexed by arbitrary byte tokens."""

    embeddings: np.ndarray
    token_bytes: np.ndarray
    token_offsets: np.ndarray
    counts: np.ndarray
    manifest: dict[str, Any]

    def __post_init__(self) -> None:
        self._rows = {
            bytes(
                self.token_bytes[self.token_offsets[i] : self.token_offsets[i + 1]]
            ): i
            for i in range(self.counts.size)
        }

    def row(self, token: bytes) -> int | None:
        return self._rows.get(token)

    def vector(self, token: bytes) -> np.ndarray | None:
        row = self.row(token)
        return None if row is None else self.embeddings[row]


def load_lookup_artifact(
    path: Path, *, expected: dict[str, str] | None = None
) -> LookupEmbeddings:
    manifest = verify_artifact(path, expected=expected, format_name=LOOKUP_FORMAT)
    arrays = {name: _load_array(path / name, mmap=True) for name in _LOOKUP_ARRAYS}
    embeddings = arrays["input_embeddings.npy"]
    offsets = arrays["token_offsets.npy"]
    counts = arrays["counts.npy"]
    token_bytes = arrays["token_bytes.npy"]
    if (
        embeddings.ndim != 2
        or embeddings.dtype != np.float32
        or counts.ndim != 1
        or offsets.shape != (counts.size + 1,)
        or int(offsets[0]) != 0
        or int(offsets[-1]) != token_bytes.size
        or embeddings.shape[0] != counts.size
    ):
        raise ValueError("invalid W2V lookup array structure")
    return LookupEmbeddings(embeddings, token_bytes, offsets, counts, manifest)


def create_checkpoint_manager(
    root: Path,
    *,
    session: Any,
    resource_version: str,
    policy: CheckpointRetentionPolicy | None = None,
) -> CheckpointManager:
    """Bind an engine session to the generic generation/pointer manager."""
    resolved_policy = policy or CheckpointRetentionPolicy()
    if resolved_policy.best_payload != "full":
        raise ValueError("W2V best checkpoints must retain full resumable state")
    return CheckpointManager(
        root=root,
        config_digest=session.export_state().config_digest,
        save_fn=lambda path, payload: save_checkpoint(
            session.export_state(),
            path,
            resource_version=resource_version,
            payload=payload,
        ),
        epoch_fn=lambda: session.completed_epochs,
        step_fn=lambda: session.export_state().processed_tokens,
        policy=resolved_policy,
    )


def verify_artifact(
    path: Path,
    *,
    expected: dict[str, str] | None = None,
    format_name: str | None = None,
) -> dict[str, Any]:
    path = Path(path)
    manifest = _read_json(path / "manifest.json")
    if manifest.get("format") != format_name or manifest.get("format_version") != 1:
        raise ValueError("unsupported W2V artifact format")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError("invalid W2V artifact manifest")
    actual_files = {item.name for item in path.iterdir() if item.is_file()}
    if actual_files != {*files, "manifest.json"}:
        raise ValueError("W2V artifact file set does not match its manifest")
    for name, metadata in files.items():
        file_path = path / name
        if _sha256(file_path) != metadata.get("sha256"):
            raise ValueError(f"W2V artifact digest mismatch: {name}")
        if name.endswith(".npy"):
            value = _load_array(file_path, mmap=True)
            if list(value.shape) != metadata.get(
                "shape"
            ) or value.dtype.str != metadata.get("dtype"):
                raise ValueError(f"W2V artifact array metadata mismatch: {name}")
    for key, value in (expected or {}).items():
        if manifest.get(key) != value:
            raise ValueError(f"W2V artifact identity mismatch: {key}")
    return manifest


def _write_manifest(
    path: Path, format_name: str, identity: dict[str, Any], array_names: tuple[str, ...]
) -> None:
    files: dict[str, dict[str, Any]] = {}
    for name in (
        *array_names,
        *(("trainer_state.json",) if format_name == CHECKPOINT_FORMAT else ()),
    ):
        metadata: dict[str, Any] = {"sha256": _sha256(path / name)}
        if name.endswith(".npy"):
            value = _load_array(path / name, mmap=True)
            metadata.update(dtype=value.dtype.str, shape=list(value.shape))
        files[name] = metadata
    _write_json(
        path / "manifest.json",
        {"format": format_name, "format_version": 1, **identity, "files": files},
    )


def _validate_manifest_identity(
    manifest: dict[str, Any], trainer: dict[str, Any]
) -> None:
    keys = (
        "schema_version",
        "config_digest",
        "vocabulary_digest",
        "corpus_digest",
        "resource_version",
        "completed_epochs",
        "processed_tokens",
    )
    if any(manifest.get(key) != trainer.get(key) for key in keys):
        raise ValueError("checkpoint manifest and trainer state identity differ")


def _worker_record(worker: WorkerState) -> dict[str, int | float]:
    return {
        "worker_id": worker.worker_id,
        "local_token_count": worker.local_token_count,
        "last_learning_rate_update_count": worker.last_learning_rate_update_count,
        "learning_rate": worker.learning_rate,
        "window_rng_state": worker.window_rng_state,
        "subsampling_rng_state": worker.subsampling_rng_state,
        "negative_rng_state": worker.negative_rng_state,
        "objective_count": worker.objective_count,
    }


def _load_array(path: Path, *, mmap: bool = False) -> np.ndarray:
    return np.load(path, allow_pickle=False, mmap_mode="r" if mmap else None)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path.name}")
    return value


__all__ = [
    "CHECKPOINT_FORMAT",
    "LOOKUP_FORMAT",
    "LookupEmbeddings",
    "create_checkpoint_manager",
    "load_checkpoint",
    "load_lookup_artifact",
    "save_checkpoint",
    "save_lookup_artifact",
    "verify_artifact",
]
