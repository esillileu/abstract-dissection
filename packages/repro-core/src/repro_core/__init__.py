"""Reproduction Core - Infrastructure for Deep Learning Paper Reproductions."""

from .context import ExperimentContext, RunContext, RuntimePaths, WorkspacePaths
from .numerics import Backend, BackendConfig, make_backend, resolve_backend
from .registry import (
    CommandGroups,
    DomainRegistry,
    StudyPlugin,
    get_executor,
    register_executor,
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
    "get_executor",
    "make_backend",
    "register_executor",
    "resolve_backend",
]
