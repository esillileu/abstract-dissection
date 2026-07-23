"""State-aware extension of the ordinary sequential container."""

from __future__ import annotations

from .sequential import Sequential


class TimeSequential(Sequential):
    """A ``Sequential`` that exposes recurrent-state lifecycle operations."""
