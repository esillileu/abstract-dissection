"""Validated W2V checkpoint serialization, retention, and inspection adapters."""

from __future__ import annotations

import errno
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from repro_core.context.checkpoint import (
    CheckpointManager,
    CheckpointRetentionPolicy,
)
from w2v import (
    TrainingSession,
    TrainingState,
    VocabularyState,
    WorkerState,
)

from .formats import (
    _CHECKPOINT_ARRAYS,
    _VOCABULARY_ARRAYS,
    CHECKPOINT_FORMAT,
    _load_array,
    _read_json,
    _write_json,
    _write_manifest,
    verify_artifact,
)

MATRIX_CHUNK_ELEMENTS = 1 << 20


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


def save_session_checkpoint(
    session: TrainingSession,
    path: Path,
    *,
    payload: str = "full",
    history: Path | None = None,
    vocabulary_artifact: Path | None = None,
) -> None:
    """Serialize at epoch boundaries with one bounded matrix chunk in RAM."""
    if payload != "full":
        raise ValueError("W2V checkpoints require full state")
    path.mkdir(parents=True, exist_ok=False)
    schema, config, vocabulary_digest, corpus = session.descriptor()
    if vocabulary_artifact is None:
        vocabulary = session.vocabulary_state()
        vocabulary_identity = dict(
            hash_capacity=vocabulary.hash_capacity,
            retained_token_count=vocabulary.retained_token_count,
        )
        for name, value in zip(_VOCABULARY_ARRAYS, vocabulary.arrays(), strict=True):
            np.save(path / name, value, allow_pickle=False)
    else:
        from .formats import VOCABULARY_FORMAT

        metadata = verify_artifact(
            vocabulary_artifact,
            format_name=VOCABULARY_FORMAT,
            expected={"corpus_digest": corpus, "vocabulary_digest": vocabulary_digest},
        )
        vocabulary_identity = {
            key: metadata[key] for key in ("hash_capacity", "retained_token_count")
        }
        for name in _VOCABULARY_ARRAYS:
            _link_or_copy(vocabulary_artifact / name, path / name)
    for output, name in (
        (False, "input_embeddings.npy"),
        (True, "output_embeddings.npy"),
    ):
        with (path / name).open("wb") as stream:
            np.lib.format.write_array_header_2_0(
                stream,
                dict(
                    descr=np.dtype(np.float32).str,
                    fortran_order=False,
                    shape=session.shape,
                ),
            )
            elements = session.shape[0] * session.shape[1]
            for start in range(0, elements, MATRIX_CHUNK_ELEMENTS):
                count = min(MATRIX_CHUNK_ELEMENTS, elements - start)
                chunk = session.embedding_chunk(output, start, count)
                stream.write(memoryview(chunk).cast("B"))
    trainer = dict(
        schema_version=schema,
        config_digest=config,
        vocabulary_digest=vocabulary_digest,
        corpus_digest=corpus,
        completed_epochs=session.completed_epochs,
        processed_tokens=session.processed_tokens,
        vocabulary=vocabulary_identity,
        workers=[_worker_record(worker) for worker in session.workers()],
    )
    _write_json(path / "trainer_state.json", trainer)
    names = _CHECKPOINT_ARRAYS
    if history is not None:
        for name in ("observations.csv", "epoch_reports.json"):
            source = history.parent / name
            if source.is_file():
                shutil.copyfile(source, path / name)
                names = (*names, name)
    _write_manifest(path, CHECKPOINT_FORMAT, trainer, names)


def lookup_from_checkpoint(checkpoint: Path, target: Path) -> None:
    """Reuse final checkpoint arrays through hard links on the same filesystem."""
    from .formats import _LOOKUP_ARRAYS
    from .lookup import LOOKUP_FORMAT

    manifest = verify_artifact(checkpoint, format_name=CHECKPOINT_FORMAT)
    target.mkdir(parents=True, exist_ok=False)
    for name in _LOOKUP_ARRAYS:
        _link_or_copy(checkpoint / name, target / name)
    identity = {
        key: manifest[key]
        for key in (
            "schema_version",
            "config_digest",
            "vocabulary_digest",
            "corpus_digest",
        )
    }
    identity["completed_epochs"] = manifest["completed_epochs"]
    identity["processed_tokens"] = manifest["processed_tokens"]
    _write_manifest(target, LOOKUP_FORMAT, identity, _LOOKUP_ARRAYS)


def create_checkpoint_manager(
    root: Path,
    *,
    session: Any,
    policy: CheckpointRetentionPolicy | None = None,
    vocabulary_artifact: Path | None = None,
) -> CheckpointManager:
    """Bind an engine session to the generic generation/pointer manager."""
    resolved_policy = policy or CheckpointRetentionPolicy()
    if resolved_policy.best_payload != "full":
        raise ValueError("W2V best checkpoints must retain full resumable state")
    return CheckpointManager(
        root=root,
        config_digest=session.descriptor()[1]
        if isinstance(session, TrainingSession)
        else session.export_state().config_digest,
        save_fn=(
            lambda path, payload: save_session_checkpoint(
                session,
                path,
                payload=payload,
                history=root.parent / "metrics/observations.csv",
                vocabulary_artifact=vocabulary_artifact,
            )
        )
        if isinstance(session, TrainingSession)
        else (
            lambda path, payload: save_checkpoint(
                session.export_state(), path, payload=payload
            )
        ),
        epoch_fn=lambda: session.completed_epochs,
        step_fn=lambda: (
            session.processed_tokens
            if isinstance(session, TrainingSession)
            else session.export_state().processed_tokens
        ),
        policy=resolved_policy,
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


def _link_or_copy(source: Path, target: Path) -> None:
    try:
        os.link(source, target)
    except OSError as exc:
        if exc.errno != errno.EXDEV:
            raise
        shutil.copyfile(source, target)


__all__ = [
    "CHECKPOINT_FORMAT",
    "create_checkpoint_manager",
    "load_checkpoint",
    "lookup_from_checkpoint",
    "save_checkpoint",
    "save_session_checkpoint",
    "verify_artifact",
]
