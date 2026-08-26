"""Dataset loader and corpus slice adapters for DS2."""

from __future__ import annotations

from typing import Any

import numpy as np
from deepscratch.datasets import load_ptb, load_sequence

from repro_core.context.paths import RuntimePaths


def load_ds2_ptb() -> dict[str, Any]:
    """Load PTB dataset using DeepScratch."""
    return load_ptb()


def load_ds2_word2vec_corpus(
    dataset: dict[str, object],
) -> tuple[np.ndarray, dict[str, int]]:
    """Load PTB or the book's fixed toy sentence for Word2Vec experiments."""
    if str(dataset.get("id")) != "DS-TOY-W2V":
        ptb = load_ptb()
        return ptb["train"], ptb["word_to_id"]
    text = str(dataset.get("text", "You say goodbye and I say hello."))
    words = text.lower().replace(".", " .").split()
    word_to_id = {word: index for index, word in enumerate(dict.fromkeys(words))}
    return np.asarray([word_to_id[word] for word in words], dtype=np.int64), word_to_id


def load_ds2_sequence(
    file_name: str,
    *,
    seed: int,
    split_algorithm: str = "default_rng",
) -> dict[str, Any]:
    """Load sequence dataset with canonical repository path injection."""
    data_path = RuntimePaths.from_environment().dataset("sequence") / file_name
    return load_sequence(
        str(data_path),
        seed=seed,
        split_algorithm=split_algorithm,
    )


def contexts_targets(corpus: Any, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate contexts and targets sliding windows for Word2Vec."""
    width = 2 * window + 1
    windows = np.lib.stride_tricks.sliding_window_view(corpus, width)
    contexts = np.concatenate((windows[:, :window], windows[:, window + 1 :]), axis=1)
    return contexts, corpus[window:-window]


def language_model_training_corpus(
    corpus: Any, dataset: dict[str, object]
) -> tuple[Any, int]:
    """Resolve the training slice and its source-compatible vocabulary size."""
    train_limit = int(dataset.get("train_limit", len(corpus)))
    train_corpus = corpus[:train_limit]
    if len(train_corpus) < 2:
        raise ValueError("language-model corpus must contain at least two tokens")
    return train_corpus, int(np.max(train_corpus)) + 1
