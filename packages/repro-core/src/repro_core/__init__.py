"""Reproduction Core - Infrastructure for Deep Learning Paper Reproductions."""

from .context import ExperimentContext, RunContext, RuntimePaths, WorkspacePaths
from .numerics import Backend, BackendConfig, make_backend, resolve_backend
from .registry import (
    CommandGroups,
    DomainRegistry,
    StudyPlugin,
)

__all__ = [
    "Backend",
    "BackendConfig",
    "CommandGroups",
    "DomainRegistry",
    "ExperimentContext",
    "RunContext",
    "RuntimePaths",
    "StudyPlugin",
    "WorkspacePaths",
    "make_backend",
    "resolve_backend",
]
