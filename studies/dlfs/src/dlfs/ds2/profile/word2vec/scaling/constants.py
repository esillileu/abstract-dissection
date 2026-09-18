from __future__ import annotations

from dlfs.ds2.profile.paths import profile_measurements

DEFAULT_RESULTS = profile_measurements("e11")
DEFAULT_VOCAB_SIZES = (
    1_000,
    2_000,
    5_000,
    10_000,
    25_000,
    50_000,
    100_000,
)
DEFAULT_CPU_VOCAB_SIZES = (
    1_000,
    2_000,
    5_000,
    10_000,
    20_000,
    50_000,
)
CONDITIONS = (
    "implemented-cbow-fs",
    "implemented-cbow-ns",
    "implemented-cbow-fused-ns",
    "implemented-skipgram-fs",
    "implemented-skipgram-ns",
    "implemented-skipgram-fused-ns",
)
EMBEDDING_SIZE = 100
CONTEXT_WIDTH = 10
NEGATIVE_SAMPLES = 5
_T_975 = (
    0.0,
    12.706,
    4.303,
    3.182,
    2.776,
    2.571,
    2.447,
    2.365,
    2.306,
    2.262,
    2.228,
    2.201,
    2.179,
    2.160,
    2.145,
    2.131,
    2.120,
    2.110,
    2.101,
    2.093,
    2.086,
    2.080,
    2.074,
    2.069,
    2.064,
    2.060,
    2.056,
    2.052,
    2.048,
    2.045,
    2.042,
)


def _default_vocab_sizes(device: str) -> tuple[int, ...]:
    return (
        DEFAULT_CPU_VOCAB_SIZES
        if not device.startswith("cuda:")
        else DEFAULT_VOCAB_SIZES
    )


def default_vocabulary_sizes(device: str) -> tuple[int, ...]:
    """Return the declared device-specific vocabulary-size schedule."""
    return _default_vocab_sizes(device)


def _validate_vocab_sizes(vocab_sizes: tuple[int, ...]) -> None:
    if not vocab_sizes or min(vocab_sizes) < 2:
        raise ValueError("vocabulary sizes must contain integers of at least 2")
    if len(set(vocab_sizes)) != len(vocab_sizes):
        raise ValueError("vocabulary sizes must not contain duplicates")
