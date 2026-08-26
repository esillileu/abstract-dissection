"""Public benchmark adapter for the DS2 original variant."""

from __future__ import annotations


def build_word2vec_full_softmax(
    kind: str,
    vocab_size: int,
    hidden_size: int,
    window_size: int,
    *,
    one_hot: bool = False,
):
    """Build the original full-softmax workload through a stable adapter."""
    from .run.e02 import build_full_softmax_model

    return build_full_softmax_model(
        kind,
        vocab_size,
        hidden_size,
        window_size,
        one_hot=one_hot,
    )
