"""Preparation routines for vocabulary, phrase models, and training inputs."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from w2v import (
    Vocabulary,
    VocabularyConfig,
    VocabularyState,
)

from .formats import (
    _VOCABULARY_ARRAYS,
    VOCABULARY_FORMAT,
    _load_array,
    _write_manifest,
    verify_artifact,
)


def _vocabulary_policy(config_values: dict, full_tokens: bool) -> dict:
    semantic = {
        "min_count": int(config_values.get("min_count", 5)),
        "max_lexical_words": int(config_values.get("max_lexical_words", 0)),
        "hash_capacity": int(config_values.get("hash_capacity", 30_000_000)),
        "semantics_version": 1,
    }
    if full_tokens:
        semantic["full_tokens"] = True
    return semantic


def _vocabulary_root(cache_root: Path) -> Path:
    p = Path(cache_root)
    if p.name == "vocabulary":
        return p
    if p.name == "prepared" or (p / "vocabulary").is_dir():
        return p / "vocabulary"
    return p / "f2/w2v/vocabulary"


def vocabulary_cache_path(
    corpus_digest: str, config_values: dict, *, cache_root: Path, full_tokens: bool
) -> Path:
    semantic = _vocabulary_policy(config_values, full_tokens)
    identity = _json_digest(
        {
            "corpus_digest": corpus_digest,
            "vocabulary_config_digest": _json_digest(semantic),
        }
    )
    return _vocabulary_root(cache_root) / identity


def resolve_or_build_vocabulary(
    corpus: Any,
    config_values: dict[str, Any],
    *,
    cache_root: Path,
    corpus_digest: str | None = None,
) -> Any:
    """Load a verified shared vocabulary or build it once atomically."""
    corpus_digest = corpus_digest or corpus.digest()
    semantic_config = _vocabulary_policy(config_values, corpus.full_tokens)
    config_digest = _json_digest(semantic_config)
    identity = _json_digest(
        {"corpus_digest": corpus_digest, "vocabulary_config_digest": config_digest}
    )
    target = _vocabulary_root(cache_root) / identity
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
    del vocabulary
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
        verify_artifact(temporary, expected=identity, format_name=VOCABULARY_FORMAT)

        _publish_atomic_directory(
            temporary, target, expected=identity, format_name=VOCABULARY_FORMAT
        )
        temporary = None
    finally:
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary)


def _publish_atomic_directory(
    temporary: Path, target: Path, *, expected: dict[str, Any], format_name: str
) -> None:
    for attempt in range(2):
        try:
            os.replace(temporary, target)
            return
        except OSError:
            try:
                verify_artifact(target, expected=expected, format_name=format_name)
            except (FileNotFoundError, OSError, ValueError):
                if attempt == 1:
                    raise
                if target.is_dir():
                    shutil.rmtree(target)
                elif target.exists():
                    target.unlink()
            else:
                return


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


__all__ = [
    "VOCABULARY_FORMAT",
    "resolve_or_build_vocabulary",
    "vocabulary_cache_path",
]
