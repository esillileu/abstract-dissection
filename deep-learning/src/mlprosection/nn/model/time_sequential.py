"""State-aware extension of the ordinary sequential container."""

from __future__ import annotations

from .sequential import Sequential


class TimeSequential(Sequential):
    """A ``Sequential`` that exposes recurrent-state lifecycle operations."""

    def reset_state(self) -> None:
        for layer in self.layers:
            reset = getattr(layer, "reset_state", None)
            if reset is not None:
                reset()

    def detach_state(self) -> None:
        for layer in self.layers:
            detach = getattr(layer, "detach_state", None)
            if detach is not None:
                detach()
