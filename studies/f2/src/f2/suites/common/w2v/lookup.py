"""Validated W2V read-only lookup artifact adapters and embeddings index."""

from __future__ import annotations

import sqlite3
import tempfile
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .formats import (
    _LOOKUP_ARRAYS,
    CHECKPOINT_FORMAT,
    LOOKUP_FORMAT,
    _load_array,
    _read_json,
    _write_manifest,
    verify_artifact,
)


@dataclass
class LookupEmbeddings:
    """Memory-mapped input embeddings indexed by arbitrary byte tokens."""

    embeddings: np.ndarray
    token_bytes: np.ndarray
    token_offsets: np.ndarray
    counts: np.ndarray
    manifest: dict[str, Any]

    def __post_init__(self) -> None:
        self._index: sqlite3.Connection | None = None

    def row(self, token: bytes) -> int | None:
        if self._index is None:
            from repro_core.context import RuntimePaths

            root = RuntimePaths.from_environment().cache_root / "f2/w2v/lookup-index"
            root.mkdir(parents=True, exist_ok=True)
            directory = tempfile.TemporaryDirectory(dir=root)
            connection = sqlite3.connect(
                Path(directory.name) / "rows.sqlite", check_same_thread=False
            )
            try:
                connection.execute("PRAGMA cache_size=-16384")
                connection.execute("PRAGMA journal_mode=OFF")
                connection.execute(
                    "CREATE TABLE rows (token BLOB PRIMARY KEY, row INTEGER) WITHOUT ROWID"
                )
                connection.executemany(
                    "INSERT OR IGNORE INTO rows VALUES (?,?)",
                    (
                        (
                            bytes(
                                self.token_bytes[
                                    self.token_offsets[i] : self.token_offsets[i + 1]
                                ]
                            ),
                            i,
                        )
                        for i in range(self.counts.size)
                    ),
                )
                connection.commit()
            except BaseException:
                connection.close()
                directory.cleanup()
                raise
            self._index = connection
            weakref.finalize(self, _close_lookup_index, connection, directory)
        result = self._index.execute(
            "SELECT row FROM rows WHERE token=?", (token,)
        ).fetchone()
        return None if result is None else int(result[0])

    def vector(self, token: bytes) -> np.ndarray | None:
        row = self.row(token)
        return None if row is None else self.embeddings[row]


def _close_lookup_index(
    connection: sqlite3.Connection, directory: tempfile.TemporaryDirectory
) -> None:
    connection.close()
    directory.cleanup()


def save_lookup_artifact(state: Any, path: Path) -> None:
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


def load_lookup_artifact(
    path: Path, *, expected: dict[str, str] | None = None
) -> LookupEmbeddings:
    format_name = _read_json(path / "manifest.json").get("format")
    if format_name not in {LOOKUP_FORMAT, CHECKPOINT_FORMAT}:
        raise ValueError("unsupported lookup artifact format")
    manifest = verify_artifact(path, expected=expected, format_name=format_name)
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


__all__ = [
    "LOOKUP_FORMAT",
    "LookupEmbeddings",
    "load_lookup_artifact",
    "save_lookup_artifact",
]
