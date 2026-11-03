"""Compatibility re-exports for refactored W2V preparation, checkpoints, and lookup modules."""

from __future__ import annotations

from .checkpoints import (
    create_checkpoint_manager,
    load_checkpoint,
    lookup_from_checkpoint,
    save_checkpoint,
    save_session_checkpoint,
)
from .formats import (
    _CHECKPOINT_ARRAYS,
    _LOOKUP_ARRAYS,
    _VOCABULARY_ARRAYS,
    CHECKPOINT_FORMAT,
    LOOKUP_FORMAT,
    VOCABULARY_FORMAT,
    _load_array,
    _read_json,
    _write_json,
    _write_manifest,
    verify_artifact,
)
from .lookup import (
    LookupEmbeddings,
    load_lookup_artifact,
    save_lookup_artifact,
)
from .preparation import (
    resolve_or_build_vocabulary,
    vocabulary_cache_path,
)

__all__ = [
    "CHECKPOINT_FORMAT",
    "LOOKUP_FORMAT",
    "VOCABULARY_FORMAT",
    "_CHECKPOINT_ARRAYS",
    "_LOOKUP_ARRAYS",
    "_VOCABULARY_ARRAYS",
    "LookupEmbeddings",
    "_load_array",
    "_read_json",
    "_write_json",
    "_write_manifest",
    "create_checkpoint_manager",
    "load_checkpoint",
    "load_lookup_artifact",
    "lookup_from_checkpoint",
    "resolve_or_build_vocabulary",
    "save_checkpoint",
    "save_lookup_artifact",
    "save_session_checkpoint",
    "verify_artifact",
    "vocabulary_cache_path",
]
