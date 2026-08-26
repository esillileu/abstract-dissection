"""Batch representation adapters for Word2Vec models in DS2."""

from __future__ import annotations

from typing import Any

from deepscratch.nn.model.architecture import (
    CBOWBatchAdapter,
    OneHotCBOWBatchAdapter,
    OneHotSkipGramBatchAdapter,
    PairExpandedSkipGramBatchAdapter,
    SkipGramBatchAdapter,
)


def build_word2vec_batch_adapter(
    architecture: str,
    input_representation: str,
    vocab_size: int,
    objective_name: str,
) -> Any:
    """Instantiate a DeepScratch batch adapter for Word2Vec inputs."""
    if input_representation == "one_hot" and objective_name != "SoftmaxWithLoss":
        raise ValueError(
            "one-hot Word2Vec input is only supported with SoftmaxWithLoss"
        )
    if architecture == "DumbSkipGram" and objective_name == "SoftmaxWithLoss":
        return PairExpandedSkipGramBatchAdapter()

    adapters = {
        ("CBOW", "embedding"): CBOWBatchAdapter(),
        ("DumbCBOW", "embedding"): CBOWBatchAdapter(),
        ("DumbSkipGram", "embedding"): SkipGramBatchAdapter(),
        ("FusedNegativeSamplingCBOW", "embedding"): CBOWBatchAdapter(),
        ("FusedNegativeSamplingSkipGram", "embedding"): SkipGramBatchAdapter(),
        ("SkipGram", "embedding"): SkipGramBatchAdapter(),
        ("CBOW", "one_hot"): OneHotCBOWBatchAdapter(vocab_size),
        ("SkipGram", "one_hot"): OneHotSkipGramBatchAdapter(vocab_size),
    }
    key = (architecture, input_representation)
    if key not in adapters:
        raise ValueError(
            f"unsupported Word2Vec architecture/input_representation: {architecture}/{input_representation}"
        )
    return adapters[key]
