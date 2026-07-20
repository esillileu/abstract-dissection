"""Device-resident samplers for objectives with discrete output targets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from mlprosection import Tensor
from mlprosection.core.backend import Backend, resolve_backend


@dataclass(frozen=True)
class UnigramSamplerMetadata:
    algorithm: str
    power: float
    replacement: bool
    excludes_positive: bool
    rejection_rounds: int


class UnigramSampler:
    """Alias-table sampler whose hot path stays entirely on the active device.

    The table is built once from a CPU corpus (or precomputed weights) and then
    copied to the selected backend. Sampling uses replacement, which is the
    usual efficient negative-sampling policy; the positive label is rejected.
    """

    algorithm = "alias_target_rejection_v1"

    def __init__(
        self,
        weights: Any,
        *,
        backend: Backend | str | None = None,
        power: float = 0.75,
        rejection_rounds: int = 4,
    ) -> None:
        if power <= 0:
            raise ValueError("power must be positive")
        if rejection_rounds < 0:
            raise ValueError("rejection_rounds must be non-negative")
        resolved = resolve_backend(backend)
        source = np.asarray(weights, dtype=np.float64)
        if source.ndim != 1 or len(source) < 2:
            raise ValueError("unigram sampling requires weights for at least two tokens")
        if np.any(source < 0) or not np.isfinite(source).all() or source.sum() <= 0:
            raise ValueError("unigram weights must be finite, non-negative, and non-empty")

        probabilities = source**power
        probabilities /= probabilities.sum()
        probability, alias = _build_alias_table(probabilities)
        self.backend = resolved
        self.vocab_size = len(probabilities)
        self.power = power
        self.rejection_rounds = rejection_rounds
        self.probability = resolved.asarray(probability, dtype=resolved.float_dtype)
        self.alias = resolved.asarray(alias, dtype=resolved.xp.int64)

    @classmethod
    def from_corpus(
        cls,
        corpus: Any,
        *,
        vocab_size: int,
        backend: Backend | str | None = None,
        power: float = 0.75,
        rejection_rounds: int = 4,
    ) -> UnigramSampler:
        """Build counts in one linear CPU pass, rather than scanning per token."""
        tokens = np.asarray(corpus, dtype=np.int64).reshape(-1)
        if len(tokens) == 0 or tokens.min() < 0 or tokens.max() >= vocab_size:
            raise ValueError("corpus contains invalid token indices")
        counts = np.bincount(tokens, minlength=vocab_size)
        return cls(
            counts,
            backend=backend,
            power=power,
            rejection_rounds=rejection_rounds,
        )

    @classmethod
    def uniform(
        cls,
        vocab_size: int,
        *,
        backend: Backend | str | None = None,
    ) -> UnigramSampler:
        return cls(np.ones(vocab_size), backend=backend, power=1.0)

    @property
    def metadata(self) -> dict[str, object]:
        return UnigramSamplerMetadata(
            algorithm=self.algorithm,
            power=self.power,
            replacement=True,
            excludes_positive=True,
            rejection_rounds=self.rejection_rounds,
        ).__dict__

    def sample(self, targets: Tensor | Any, *, sample_size: int):
        """Return ``(batch, sample_size)`` negatives, excluding each target.

        Rejection has a fixed number of fully vectorized rounds to avoid a
        host synchronization in the hot path. The final deterministic fallback
        guarantees target exclusion even for an unusually frequent token.
        """
        if sample_size < 1:
            raise ValueError("sample_size must be positive")
        xp = self.backend.xp
        values = targets.data if isinstance(targets, Tensor) else targets
        labels = xp.asarray(values, dtype=xp.int64).reshape(-1)

        negatives = self._draw((len(labels), sample_size))
        expected = labels[:, None]
        for _ in range(self.rejection_rounds):
            replacement = self._draw((len(labels), sample_size))
            negatives = xp.where(negatives == expected, replacement, negatives)
        fallback = (expected + 1) % self.vocab_size
        return xp.where(negatives == expected, fallback, negatives)

    def _draw(self, shape: tuple[int, int]):
        xp = self.backend.xp
        columns = xp.random.randint(self.vocab_size, size=shape)
        coin = xp.random.random(shape)
        return xp.where(coin < self.probability[columns], columns, self.alias[columns])


def _build_alias_table(probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build Vose's alias table once on the host."""
    size = len(probabilities)
    scaled = probabilities * size
    probability = np.zeros(size, dtype=np.float64)
    alias = np.zeros(size, dtype=np.int64)
    small = [int(index) for index in np.flatnonzero(scaled < 1.0)]
    large = [int(index) for index in np.flatnonzero(scaled >= 1.0)]

    while small and large:
        lower = small.pop()
        higher = large.pop()
        probability[lower] = scaled[lower]
        alias[lower] = higher
        scaled[higher] -= 1.0 - scaled[lower]
        if scaled[higher] < 1.0:
            small.append(higher)
        else:
            large.append(higher)

    for index in small + large:
        probability[index] = 1.0
        alias[index] = index
    return probability, alias
