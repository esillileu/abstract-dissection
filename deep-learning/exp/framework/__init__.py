"""Domain-neutral experiment framework contracts."""

from .registry import DomainPlugin, DomainRegistry
from .state import StateCoordinate, StateOwner, WorkspacePaths

__all__ = ["DomainPlugin", "DomainRegistry", "StateCoordinate", "StateOwner", "WorkspacePaths"]
