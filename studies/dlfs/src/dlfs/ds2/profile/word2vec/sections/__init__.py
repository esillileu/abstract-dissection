"""Expose independently measurable Word2Vec workload sections."""

from .cli import main
from .fixtures import (
    ComponentFixture,
    FusedComponentFixture,
    OriginalComponentFixture,
)
from .runner import (
    COMPONENTS,
    DEFAULT_OUTPUT,
    FUSED_COMPONENTS,
    IMPLEMENTED_COMPONENTS,
    ORIGINAL_COMPONENTS,
    profile_modules,
)

__all__ = [
    "COMPONENTS",
    "DEFAULT_OUTPUT",
    "FUSED_COMPONENTS",
    "IMPLEMENTED_COMPONENTS",
    "ORIGINAL_COMPONENTS",
    "ComponentFixture",
    "FusedComponentFixture",
    "OriginalComponentFixture",
    "main",
    "profile_modules",
]
