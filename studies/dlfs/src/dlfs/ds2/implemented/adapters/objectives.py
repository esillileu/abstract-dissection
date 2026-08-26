"""Objective translation adapters for DS2."""

from __future__ import annotations

from typing import Any

from deepscratch.nn.objective import (
    FusedNegativeSampling,
    NegativeSampling,
    SoftmaxWithLoss,
    TemporalSoftmaxCrossEntropy,
)
from deepscratch.nn.sampling import UnigramSampler


def build_unigram_sampler(
    sampler_values: dict[str, object],
    corpus: Any,
    vocab_size: int,
    backend: Any,
    rng: Any,
) -> UnigramSampler:
    """Instantiate a UnigramSampler for NegativeSampling objectives."""
    return UnigramSampler.from_corpus(
        corpus,
        vocab_size=vocab_size,
        backend=backend,
        power=float(sampler_values.get("power", 0.75)),
        rejection_rounds=int(sampler_values.get("rejection_rounds", 4)),
        algorithm=str(sampler_values.get("algorithm", UnigramSampler.ALIAS_REJECTION)),
        rng=rng,
    )


def build_word2vec_objective(
    objective_name: str,
    objective_config: dict[str, object],
    vocab_size: int,
    sampler: UnigramSampler | None,
    backend: Any,
    *,
    is_skipgram: bool = False,
) -> Any:
    """Instantiate a DeepScratch Word2Vec objective from config."""
    if objective_name == "SoftmaxWithLoss":
        return SoftmaxWithLoss(
            reduction=str(objective_config.get("reduction", "mean")),
            grouped_targets=is_skipgram,
            backend=backend,
        )
    if objective_name in {"NegativeSampling", "FusedNegativeSampling"}:
        objective_type = (
            FusedNegativeSampling
            if objective_name == "FusedNegativeSampling"
            else NegativeSampling
        )
        return objective_type(
            vocab_size,
            negative_samples=int(objective_config.get("negative_samples", 5)),
            reduction=str(objective_config.get("reduction", "mean")),
            sampler=sampler,
            backend=backend,
        )
    raise ValueError(f"unknown Word2Vec objective name: {objective_name}")


def build_sequence_objective(
    objective_config: dict[str, object],
    backend: Any,
) -> TemporalSoftmaxCrossEntropy:
    """Instantiate a TemporalSoftmaxCrossEntropy objective for sequence models."""
    return TemporalSoftmaxCrossEntropy(
        reduction=str(objective_config.get("reduction", "mean")),
        backend=backend,
    )
