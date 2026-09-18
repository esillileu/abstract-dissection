"""Deterministic text normalization matching word2vec preprocessing conventions."""

from __future__ import annotations

_NORMALIZE_REPLACEMENTS = (
    ("\u2019", "'"),
    ("\u2032", "'"),
    ("''", " "),
    ("'", " ' "),
    ("“", '"'),
    ("”", '"'),
    ('"', ' " '),
    (".", " . "),
    ("<br />", " "),
    (", ", " , "),
    ("(", " ( "),
    (")", " ) "),
    ("!", " ! "),
    ("?", " ? "),
    (";", " "),
    (":", " "),
    ("-", " - "),
    ("=", " "),
    ("*", " "),
    ("|", " "),
    ("«", " "),
)


def normalize_text(text: str) -> str:
    """Port of demo-train-big-model-v1.sh normalize_text under the C locale."""
    text = text.lower()
    for old, new in _NORMALIZE_REPLACEMENTS:
        text = text.replace(old, new)
    return text.translate(str.maketrans({str(number): " " for number in range(10)}))


__all__ = ["normalize_text"]
