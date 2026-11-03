"""Sentence reconstruction, segmentation, and deterministic caching for W2V2."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
from pathlib import Path

from repro_io.checksum import sha256_file

from f2.suites.common.w2v.phrases import _tokens

SENTENCE_POLICY = "terminal-punctuation-or-lf-v1"
MAX_SENTENCE_TOKENS = 65_536
MAX_TOKEN_BYTES = 4096


def sentences(source: Path):
    """Reconstruct boundaries after .!? (optional closing quotes), and at LF."""
    sentence = []
    for token in _tokens(source, max_token_bytes=MAX_TOKEN_BYTES):
        if token is not None:
            sentence.append(token)
            if len(sentence) > MAX_SENTENCE_TOKENS:
                raise ValueError(
                    "sentence exceeds 65536 raw tokens; splitting is forbidden"
                )
        if token is None or token.rstrip(b"\"'\xe2\x80\x9d)]}").endswith(
            (b".", b"!", b"?")
        ):
            if sentence:
                yield sentence
                sentence = []
    if sentence:
        yield sentence


def sentence_cache_path(cache_root: Path, source_sha256: str) -> Path:
    key = hashlib.sha256(
        f"{source_sha256}:{SENTENCE_POLICY}:{MAX_SENTENCE_TOKENS}:{MAX_TOKEN_BYTES}".encode()
    ).hexdigest()
    return cache_root / "f2/w2v/sentences" / f"{key}.txt"


def cached_sentences(source: Path, cache_root: Path, source_sha256: str) -> Path:
    if sha256_file(source) != source_sha256:
        raise ValueError("source SHA256 mismatch before sentence preparation")
    root = cache_root / "f2/w2v/sentences"
    root.mkdir(parents=True, exist_ok=True)
    with (root / f"{source_sha256}.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _cached_sentences(source, cache_root, source_sha256)


def _cached_sentences(source: Path, cache_root: Path, source_sha256: str) -> Path:
    target = sentence_cache_path(cache_root, source_sha256)
    root = target.parent
    root.mkdir(parents=True, exist_ok=True)
    manifest = target.with_suffix(".json")
    if target.is_file() and manifest.is_file():
        try:
            data = json.loads(manifest.read_text())
            if (
                data.get("source_sha256") == source_sha256
                and data.get("policy") == SENTENCE_POLICY
                and data.get("sha256") == sha256_file(target)
            ):
                return target
        except (ValueError, OSError):
            pass
    for abandoned in root.glob(f".{target.stem}-*"):
        abandoned.unlink()
    descriptor, name = tempfile.mkstemp(prefix=f".{target.stem}-", dir=root)
    try:
        with os.fdopen(descriptor, "wb") as output:
            for sentence in sentences(source):
                output.write(b" ".join(sentence) + b"\n")
        if sha256_file(source) != source_sha256:
            raise ValueError("source changed during sentence preparation")
        os.replace(name, target)
        manifest.write_text(
            json.dumps(
                dict(
                    source_sha256=source_sha256,
                    sha256=sha256_file(target),
                    policy=SENTENCE_POLICY,
                ),
                sort_keys=True,
            )
            + "\n"
        )
    finally:
        Path(name).unlink(missing_ok=True)
    return target


__all__ = [
    "MAX_SENTENCE_TOKENS",
    "MAX_TOKEN_BYTES",
    "SENTENCE_POLICY",
    "cached_sentences",
    "sentence_cache_path",
    "sentences",
]
