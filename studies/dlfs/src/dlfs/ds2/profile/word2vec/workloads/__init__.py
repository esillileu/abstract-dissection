"""Construct original and implemented Word2Vec profiling workloads."""

from __future__ import annotations

from .condition import (
    _batch,
    _build_condition,
    _run_updates,
    _runtime_estimates,
    build_profile_condition,
    profile_batch,
)
from .constants import (
    BOOK_ROOT,
    CONDITIONS,
    DEFAULT_EPOCHS,
    DEFAULT_OUTPUT,
    PTB_TRAIN,
    ROOT,
    STAGES,
)
from .env import (
    _install_original_imports,
    _load_data,
    _metadata,
    _phase,
    load_profile_data,
    profile_metadata,
)
from .implemented import (
    ImplementedWord2Vec,
    _implemented_components,
    run_fused_update,
    run_implemented_update,
)
from .original import OriginalWord2Vec
from .runner import (
    ConditionResult,
    _stage_value,
    main,
    profile_condition,
)

__all__ = [
    "BOOK_ROOT",
    "CONDITIONS",
    "DEFAULT_EPOCHS",
    "DEFAULT_OUTPUT",
    "PTB_TRAIN",
    "ROOT",
    "STAGES",
    "ConditionResult",
    "ImplementedWord2Vec",
    "OriginalWord2Vec",
    "_batch",
    "_build_condition",
    "_implemented_components",
    "_install_original_imports",
    "_load_data",
    "_metadata",
    "_phase",
    "_run_updates",
    "_runtime_estimates",
    "_stage_value",
    "build_profile_condition",
    "load_profile_data",
    "main",
    "profile_batch",
    "profile_condition",
    "profile_metadata",
    "run_fused_update",
    "run_implemented_update",
]
