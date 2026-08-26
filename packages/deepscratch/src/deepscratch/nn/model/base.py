"""Architecture-neutral model contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from deepscratch.nn.layers import Layer
from deepscratch.nn.layers.base import _resolve_owner


class Model(Layer):
    """Base class for trainable prediction models.

    Objectives are deliberately not part of this class.  Runtime state is the
    subset of registered buffers marked ``runtime_state=True``.
    """

    def snapshot_runtime_state(self) -> dict[str, Any]:
        return {
            name: None if value is None else value.copy()
            for name, value in self.named_buffers(runtime_state=True)
        }

    def restore_runtime_state(self, state: dict[str, Any]) -> None:
        current = dict(self.named_buffers(runtime_state=True))
        if set(state) != set(current):
            missing = sorted(set(current) - set(state))
            unexpected = sorted(set(state) - set(current))
            raise KeyError(
                f"runtime-state mismatch: missing={missing}, unexpected={unexpected}"
            )
        for name, value in state.items():
            owner, attr = _resolve_owner(self, name)
            setattr(owner, attr, None if value is None else value.copy())

    def reset_runtime_state(self) -> None:
        for name, _ in tuple(self.named_buffers(runtime_state=True)):
            owner, attr = _resolve_owner(self, name)
            setattr(owner, attr, None)

    def detach_runtime_state(self) -> None:
        for name, value in tuple(self.named_buffers(runtime_state=True)):
            if value is not None:
                owner, attr = _resolve_owner(self, name)
                setattr(owner, attr, value.copy())


class GenerativeModel(Model, ABC):
    """Explicit capability required by sequence-generation trainers."""

    @abstractmethod
    def generate_device(self, xs, start_id: int, sample_size: int):
        raise NotImplementedError
