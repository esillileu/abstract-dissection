"""Device-resident samplers for objectives with discrete output targets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from deepscratch.core import Tensor
from deepscratch.core.backend import Backend, resolve_backend


@dataclass(frozen=True)
class UnigramSamplerMetadata:
    algorithm: str
    power: float
    replacement: bool
    excludes_positive: bool
    rejection_rounds: int | None


class UnigramSampler:
    """Device-resident unigram sampler with exact target exclusion available.

    Distribution tables are built once from CPU weights and copied to the
    selected backend. The conditional-CDF algorithm removes the positive
    target's probability interval before drawing, so it needs neither repeated
    rejection draws nor a biased fallback.
    """

    ALIAS_REJECTION = "alias_target_rejection_v1"
    CONDITIONAL_CDF = "conditional_cdf_target_exclusion_v1"

    def __init__(
        self,
        weights: Any,
        *,
        backend: Backend | str | None = None,
        power: float = 0.75,
        rejection_rounds: int = 4,
        algorithm: str = ALIAS_REJECTION,
        rng=None,
    ) -> None:
        if power <= 0:
            raise ValueError("power must be positive")
        if rejection_rounds < 0:
            raise ValueError("rejection_rounds must be non-negative")
        if algorithm not in {self.ALIAS_REJECTION, self.CONDITIONAL_CDF}:
            raise ValueError(f"unsupported unigram sampling algorithm: {algorithm}")
        resolved = resolve_backend(backend)
        source = np.asarray(weights, dtype=np.float64)
        if source.ndim != 1 or len(source) < 2:
            raise ValueError(
                "unigram sampling requires weights for at least two tokens"
            )
        if np.any(source < 0) or not np.isfinite(source).all() or source.sum() <= 0:
            raise ValueError(
                "unigram weights must be finite, non-negative, and non-empty"
            )

        probabilities = source**power
        probabilities /= probabilities.sum()
        if algorithm == self.CONDITIONAL_CDF and np.count_nonzero(probabilities) < 2:
            raise ValueError(
                "conditional-CDF sampling requires at least two positive weights"
            )
        self.backend = resolved
        self.vocab_size = len(probabilities)
        self.power = power
        self.algorithm = algorithm
        self.rejection_rounds = rejection_rounds
        self.rng = rng if rng is not None else resolved.xp.random
        if algorithm == self.ALIAS_REJECTION:
            probability, alias = _build_alias_table(probabilities)
            self.probability = resolved.asarray(probability, dtype=resolved.float_dtype)
            self.alias = resolved.asarray(alias, dtype=resolved.xp.int64)
            self.distribution = None
            self.cumulative_distribution = None
        else:
            cumulative = np.cumsum(probabilities, dtype=np.float64)
            cumulative[-1] = 1.0
            self.probability = None
            self.alias = None
            self.distribution = resolved.asarray(
                probabilities, dtype=resolved.xp.float64
            )
            self.cumulative_distribution = resolved.asarray(
                cumulative, dtype=resolved.xp.float64
            )

    @classmethod
    def from_corpus(
        cls,
        corpus: Any,
        *,
        vocab_size: int,
        backend: Backend | str | None = None,
        power: float = 0.75,
        rejection_rounds: int = 4,
        algorithm: str = ALIAS_REJECTION,
        rng=None,
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
            algorithm=algorithm,
            rng=rng,
        )

    @classmethod
    def uniform(
        cls,
        vocab_size: int,
        *,
        backend: Backend | str | None = None,
        algorithm: str = ALIAS_REJECTION,
        rng=None,
    ) -> UnigramSampler:
        return cls(
            np.ones(vocab_size),
            backend=backend,
            power=1.0,
            algorithm=algorithm,
            rng=rng,
        )

    @property
    def metadata(self) -> dict[str, object]:
        return UnigramSamplerMetadata(
            algorithm=self.algorithm,
            power=self.power,
            replacement=True,
            excludes_positive=True,
            rejection_rounds=(
                self.rejection_rounds
                if self.algorithm == self.ALIAS_REJECTION
                else None
            ),
        ).__dict__

    def sample(self, targets: Tensor | Any, *, sample_size: int):
        """Return ``(batch, sample_size)`` negatives, excluding each target."""
        if sample_size < 1:
            raise ValueError("sample_size must be positive")
        xp = self.backend.xp
        values = targets.data if isinstance(targets, Tensor) else targets
        labels = xp.asarray(values, dtype=xp.int64).reshape(-1)

        if self.algorithm == self.CONDITIONAL_CDF:
            return self._sample_conditional_cdf(labels, sample_size)

        negatives = self._draw((len(labels), sample_size))
        expected = labels[:, None]
        for _ in range(self.rejection_rounds):
            replacement = self._draw((len(labels), sample_size))
            negatives = xp.where(negatives == expected, replacement, negatives)
        fallback = (expected + 1) % self.vocab_size
        return xp.where(negatives == expected, fallback, negatives)

    def _sample_conditional_cdf(self, labels, sample_size: int):
        """Sample exactly from p(word | word != target) with one random draw."""
        xp = self.backend.xp
        distribution = self.distribution
        cumulative = self.cumulative_distribution
        if distribution is None or cumulative is None:
            raise RuntimeError("conditional-CDF tables are unavailable")
        target_probability = distribution[labels][:, None]
        interval_start = (cumulative[labels] - distribution[labels])[:, None]
        draws = self.rng.random_sample((len(labels), sample_size))
        draws = draws.astype(xp.float64, copy=False) * (1.0 - target_probability)
        adjusted = draws + (draws >= interval_start) * target_probability
        return xp.searchsorted(cumulative, adjusted, side="right")

    def _draw(self, shape: tuple[int, int]):
        xp = self.backend.xp
        if self.probability is None or self.alias is None:
            raise RuntimeError("alias tables are unavailable")
        columns = self.rng.randint(self.vocab_size, size=shape)
        coin = self.rng.random_sample(shape)
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
