"""Model translation adapters for DS2."""

from __future__ import annotations

from typing import Any

from deepscratch.nn.model.architecture import (
    CBOW,
    AttentionPeekySeq2seq,
    AttentionSeq2seq,
    BetterRnnlm,
    DumbCBOW,
    DumbSkipGram,
    FusedNegativeSamplingCBOW,
    FusedNegativeSamplingSkipGram,
    OneHotCBOW,
    OneHotSkipGram,
    PeekySeq2seq,
    Rnnlm,
    Seq2seq,
    SkipGram,
    TiedRnnlm,
    VanillaRnnlm,
)


def build_word2vec_model(
    architecture: str,
    input_representation: str,
    vocab_size: int,
    embedding_size: int,
    backend: Any,
) -> Any:
    """Instantiate a DeepScratch Word2Vec model from architecture/representation names."""
    model_type = {
        ("CBOW", "embedding"): CBOW,
        ("DumbCBOW", "embedding"): DumbCBOW,
        ("DumbSkipGram", "embedding"): DumbSkipGram,
        ("FusedNegativeSamplingCBOW", "embedding"): FusedNegativeSamplingCBOW,
        ("FusedNegativeSamplingSkipGram", "embedding"): FusedNegativeSamplingSkipGram,
        ("SkipGram", "embedding"): SkipGram,
        ("CBOW", "one_hot"): OneHotCBOW,
        ("SkipGram", "one_hot"): OneHotSkipGram,
    }.get((architecture, input_representation))
    if model_type is None:
        raise ValueError(
            "unknown Word2Vec architecture/input representation: "
            f"{architecture}/{input_representation}"
        )
    return model_type(vocab_size, embedding_size, backend=backend)


def build_language_model(
    name: str,
    vocab_size: int,
    values: dict[str, object],
    backend: Any,
    *,
    dropout_rng=None,
) -> Any:
    """Instantiate a DeepScratch language model from a DS2 model configuration dictionary."""
    kwargs = {
        "vocab_size": vocab_size,
        "wordvec_size": int(values.get("wordvec_size", 100)),
        "hidden_size": int(values.get("hidden_size", 100)),
        "backend": backend,
    }
    if name == "VanillaRnnlm":
        return VanillaRnnlm(**kwargs)
    if name == "Rnnlm":
        return Rnnlm(**kwargs)
    if name == "TiedRnnlm":
        return TiedRnnlm(**kwargs)
    if name == "BetterRnnlm":
        return BetterRnnlm(
            **kwargs,
            dropout_ratio=float(values.get("dropout_ratio", 0.5)),
            dropout_rng=dropout_rng,
        )
    raise ValueError(f"unknown language-model name: {name}")


def build_seq2seq_model(
    name: str,
    vocab_size: int,
    values: dict[str, object],
    backend: Any,
) -> Any:
    """Instantiate a DeepScratch seq2seq model from a DS2 model configuration dictionary."""
    kwargs = {
        "vocab_size": vocab_size,
        "wordvec_size": int(values.get("wordvec_size", 16)),
        "hidden_size": int(values.get("hidden_size", 128)),
        "backend": backend,
    }
    if name == "Seq2seq":
        return Seq2seq(**kwargs)
    if name == "PeekySeq2seq":
        return PeekySeq2seq(**kwargs)
    if name == "AttentionSeq2seq":
        return AttentionSeq2seq(**kwargs)
    if name == "AttentionPeekySeq2seq":
        return AttentionPeekySeq2seq(**kwargs)
    raise ValueError(f"unknown seq2seq name: {name}")
