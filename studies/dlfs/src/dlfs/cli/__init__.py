"""Domain-owned CLI callbacks for DeepScratch volumes and variants."""

from __future__ import annotations

from .analysis import analyze
from .common import _selected_variant, _writer_overrides
from .execution import plan, run
from .profile import profile
from .status import check

__all__ = [
    "_selected_variant",
    "_writer_overrides",
    "analyze",
    "check",
    "plan",
    "profile",
    "run",
]
