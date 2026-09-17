"""DeepScratch state serialization and deserialization adapter for checkpoints."""

from __future__ import annotations

from .manager import (
    create_deepscratch_checkpoint_manager,
    load_deepscratch_checkpoint,
    save_deepscratch_epoch_checkpoint,
    write_deepscratch_checkpoint,
)
from .state import (
    _backend_rng_state,
    _load_buffers,
    _load_model_parameters_compatible,
    _read_pickle,
    _resolve_owner,
    _restore_backend_rng_state,
    _save_buffers,
    _write_pickle,
    load_deepscratch_model_parameters,
)

__all__ = [
    "_backend_rng_state",
    "_load_buffers",
    "_load_model_parameters_compatible",
    "_read_pickle",
    "_resolve_owner",
    "_restore_backend_rng_state",
    "_save_buffers",
    "_write_pickle",
    "create_deepscratch_checkpoint_manager",
    "load_deepscratch_checkpoint",
    "load_deepscratch_model_parameters",
    "save_deepscratch_epoch_checkpoint",
    "write_deepscratch_checkpoint",
]
