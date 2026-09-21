"""Validated W2V checkpoint and read-only lookup artifact adapters."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from repro_io.checksum import sha256_file
from w2v import (
    TrainingState,
    Vocabulary,
    VocabularyConfig,
    VocabularyState,
    WorkerState,
)

from repro_core.context.checkpoint import (
    CheckpointManager,
    CheckpointRetentionPolicy,
)

CHECKPOINT_FORMAT = "f2-w2v-checkpoint-v1"
LOOKUP_FORMAT = "f2-w2v-lookup-v1"
VOCABULARY_FORMAT = "f2-w2v-vocabulary-v1"

_VOCABULARY_ARRAYS = (
    "token_bytes.npy",
    "token_offsets.npy",
    "counts.npy",
    "huffman_offsets.npy",
    "huffman_paths.npy",
    "huffman_bits.npy",
)
_CHECKPOINT_ARRAYS = (
    "input_embeddings.npy",
    "output_embeddings.npy",
    *_VOCABULARY_ARRAYS,
)
_LOOKUP_ARRAYS = (
    "input_embeddings.npy",
    "token_bytes.npy",
    "token_offsets.npy",
    "counts.npy",
)


def resolve_or_build_vocabulary(
    corpus: Any,
    config_values: dict[str, Any],
    *,
    cache_root: Path,
    corpus_digest: str | None = None,
) -> Any:
    """Load a verified shared vocabulary or build it once atomically."""
    corpus_digest = corpus_digest or corpus.digest()
    semantic_config = {
        "min_count": int(config_values.get("min_count", 5)),
        "max_lexical_words": int(config_values.get("max_lexical_words", 0)),
        "hash_capacity": int(config_values.get("hash_capacity", 30_000_000)),
        "semantics_version": 1,
    }
    config_digest = _json_digest(semantic_config)
    identity = _json_digest(
        {"corpus_digest": corpus_digest, "vocabulary_config_digest": config_digest}
    )
    target = Path(cache_root) / "f2" / "w2v" / "vocabulary" / identity
    expected = {
        "corpus_digest": corpus_digest,
        "vocabulary_config": semantic_config,
        "vocabulary_config_digest": config_digest,
        "identity_digest": identity,
    }
    try:
        return _load_vocabulary_artifact(target, expected=expected)
    except (FileNotFoundError, OSError, ValueError):
        pass

    vocabulary = Vocabulary.build(corpus, VocabularyConfig(**config_values))
    _write_vocabulary_artifact(
        target,
        vocabulary.export_state(),
        corpus_digest=corpus_digest,
        vocabulary_config=semantic_config,
        vocabulary_config_digest=config_digest,
        identity_digest=identity,
    )
    return _load_vocabulary_artifact(target, expected=expected)


def _write_vocabulary_artifact(
    target: Path,
    state: VocabularyState,
    *,
    corpus_digest: str,
    vocabulary_config: dict[str, int],
    vocabulary_config_digest: str,
    identity_digest: str,
) -> None:
    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=parent))
    try:
        arrays = state.arrays()
        for name, value in zip(_VOCABULARY_ARRAYS, arrays, strict=True):
            np.save(temporary / name, value, allow_pickle=False)
        vocabulary_digest = state.digest()
        identity = {
            "corpus_digest": corpus_digest,
            "vocabulary_config": vocabulary_config,
            "vocabulary_config_digest": vocabulary_config_digest,
            "identity_digest": identity_digest,
            "vocabulary_digest": vocabulary_digest,
            "hash_capacity": state.hash_capacity,
            "retained_token_count": state.retained_token_count,
        }
        _write_manifest(temporary, VOCABULARY_FORMAT, identity, _VOCABULARY_ARRAYS)
        _load_vocabulary_artifact(temporary, expected=identity)

        _publish_vocabulary_artifact(temporary, target, expected=identity)
        temporary = None
    finally:
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary)


def _publish_vocabulary_artifact(
    temporary: Path, target: Path, *, expected: dict[str, Any]
) -> None:
    """Publish a verified directory, adopting a valid concurrent winner."""
    for attempt in range(2):
        try:
            os.replace(temporary, target)
            return
        except OSError:
            try:
                _load_vocabulary_artifact(target, expected=expected)
            except (FileNotFoundError, OSError, ValueError):
                if attempt == 1:
                    raise
                _remove_invalid_target(target)
            else:
                return


def _remove_invalid_target(target: Path) -> None:
    """Remove a known-invalid target only if it was not replaced meanwhile."""
    try:
        marker = target.stat()
    except FileNotFoundError:
        return
    try:
        if target.stat().st_ino != marker.st_ino:
            return
    except FileNotFoundError:
        return
    if target.is_dir():
        shutil.rmtree(target)
    else:
        target.unlink()


def _load_vocabulary_artifact(path: Path, *, expected: dict[str, Any]) -> Any:
    manifest = verify_artifact(path, expected=expected, format_name=VOCABULARY_FORMAT)
    arrays = {name: _load_array(path / name) for name in _VOCABULARY_ARRAYS}
    state = VocabularyState.from_arrays(
        arrays["token_bytes.npy"],
        arrays["token_offsets.npy"],
        arrays["counts.npy"],
        arrays["huffman_offsets.npy"],
        arrays["huffman_paths.npy"],
        arrays["huffman_bits.npy"],
        int(manifest["hash_capacity"]),
        int(manifest["retained_token_count"]),
    )
    if state.digest() != manifest.get("vocabulary_digest"):
        raise ValueError("vocabulary artifact digest mismatch")
    return Vocabulary.restore_state(state)


def _json_digest(value: Any) -> str:
    digest = hashlib.sha256()
    digest.update(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    return digest.hexdigest()


def save_checkpoint(
    state: TrainingState,
    path: Path,
    *,
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


def save_lookup_artifact(state: TrainingState, path: Path) -> None:
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
    return sha256_file(path)


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
    "VOCABULARY_FORMAT",
    "LookupEmbeddings",
    "create_checkpoint_manager",
    "load_checkpoint",
    "load_lookup_artifact",
    "resolve_or_build_vocabulary",
    "save_checkpoint",
    "save_lookup_artifact",
    "verify_artifact",
]
