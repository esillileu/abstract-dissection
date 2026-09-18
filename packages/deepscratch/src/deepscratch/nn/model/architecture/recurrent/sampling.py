"""Host and device sequence generation sampling helpers."""

from __future__ import annotations

from deepscratch.core.backend import Backend


def _stack_sampled_ids_device(backend: Backend, sampled):
    if not sampled:
        return backend.xp.empty((0, 0), dtype=backend.xp.int64)
    return backend.xp.stack(sampled, axis=1)


def _host_sampled_ids(backend: Backend, sampled) -> list[int]:
    if sampled.size == 0:
        return []
    values = backend.to_numpy(sampled)
    if values.ndim == 2:
        if values.shape[0] != 1:
            raise ValueError("generate() expects a single input sequence")
        values = values[0]
    return [int(value) for value in values]


__all__ = [
    "_host_sampled_ids",
    "_stack_sampled_ids_device",
]
