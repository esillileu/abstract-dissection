"""Shared on-disk artifact formats, manifest validation, and I/O helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from repro_io.checksum import sha256_file

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
        if sha256_file(file_path) != metadata.get("sha256"):
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
        metadata: dict[str, Any] = {"sha256": sha256_file(path / name)}
        if name.endswith(".npy"):
            value = _load_array(path / name, mmap=True)
            metadata.update(dtype=value.dtype.str, shape=list(value.shape))
        files[name] = metadata
    _write_json(
        path / "manifest.json",
        {"format": format_name, "format_version": 1, **identity, "files": files},
    )


def _load_array(path: Path, *, mmap: bool = False) -> np.ndarray:
    return np.load(path, allow_pickle=False, mmap_mode="r" if mmap else None)


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
    "_CHECKPOINT_ARRAYS",
    "_LOOKUP_ARRAYS",
    "_VOCABULARY_ARRAYS",
    "_load_array",
    "_read_json",
    "_write_json",
    "_write_manifest",
    "verify_artifact",
]
