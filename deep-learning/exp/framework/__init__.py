"""Domain-neutral experiment framework contracts."""

from .registry import CommandGroups, DomainPlugin, DomainRegistry
from .paths import StateCoordinate, StateOwner, WorkspacePaths

__all__ = [
    "CommandGroups",
    "DomainPlugin",
    "DomainRegistry",
    "StateCoordinate",
    "StateOwner",
    "WorkspacePaths",
]
