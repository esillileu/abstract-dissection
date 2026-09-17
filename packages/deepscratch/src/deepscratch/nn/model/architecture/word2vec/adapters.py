"""Explicit batch adapters and one-hot encoding utilities."""

from __future__ import annotations

from dataclasses import dataclass

from deepscratch.core import Tensor


def _one_hot(xp, indices, vocab_size: int, dtype):
    values = xp.zeros((*indices.shape, vocab_size), dtype=dtype)
    flat_values = values.reshape(-1, vocab_size)
    flat_indices = indices.reshape(-1)
    flat_values[xp.arange(len(flat_indices)), flat_indices] = 1
    return values


def _one_hot_tensor(values: Tensor, vocab_size: int) -> Tensor:
    xp = values.backend.xp
    indices = values.data.astype(xp.int64, copy=False)
    return Tensor(
        _one_hot(xp, indices, vocab_size, values.backend.float_dtype),
        backend=values.backend,
    )


@dataclass(frozen=True)
class CBOWBatchAdapter:
    def prepare(self, contexts: Tensor, targets: Tensor) -> tuple[Tensor, Tensor]:
        return contexts, targets.reshape(-1)


@dataclass(frozen=True)
class SkipGramBatchAdapter:
    def prepare(self, contexts: Tensor, targets: Tensor) -> tuple[Tensor, Tensor]:
        """Keep unique centers and group context labels for every objective."""
        return targets.reshape(-1), contexts


@dataclass(frozen=True)
class PairExpandedSkipGramBatchAdapter:
    """Expand each center-context pair into an independent prediction row."""

    def prepare(self, contexts: Tensor, targets: Tensor) -> tuple[Tensor, Tensor]:
        if contexts.ndim != 2 or targets.ndim != 1 or len(contexts) != len(targets):
            raise ValueError("Skip-gram pair expansion expects (B, C) and (B,) batches")
        xp = contexts.backend.xp
        context_count = contexts.shape[1]
        centers = xp.repeat(targets.data.reshape(-1, 1), context_count, axis=1).reshape(
            -1
        )
        return (
            Tensor(centers, backend=contexts.backend),
            contexts.reshape(-1),
        )


@dataclass(frozen=True)
class OneHotCBOWBatchAdapter:
    vocab_size: int

    def prepare(self, contexts: Tensor, targets: Tensor) -> tuple[Tensor, Tensor]:
        return (
            _one_hot_tensor(contexts, self.vocab_size),
            _one_hot_tensor(targets.reshape(-1), self.vocab_size),
        )


@dataclass(frozen=True)
class OneHotSkipGramBatchAdapter:
    vocab_size: int

    def prepare(self, contexts: Tensor, targets: Tensor) -> tuple[Tensor, Tensor]:
        return (
            _one_hot_tensor(targets.reshape(-1), self.vocab_size),
            _one_hot_tensor(contexts, self.vocab_size),
        )
