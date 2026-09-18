"""Deterministic phrase-corpus construction and lineage."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path


@dataclass(frozen=True)
class PhrasePolicy:
    passes: int
    threshold: float
    min_count: int
    separator: bytes = b"_"

    def __post_init__(self) -> None:
        if self.passes < 1 or self.threshold < 0 or self.min_count < 1:
            raise ValueError("invalid phrase detection policy")
        if not self.separator or any(byte in b" \t\r\n" for byte in self.separator):
            raise ValueError("phrase separator must be non-empty non-whitespace bytes")

    def record(self) -> dict[str, object]:
        return {
            "passes": self.passes,
            "threshold": self.threshold,
            "min_count": self.min_count,
            "separator_hex": self.separator.hex(),
        }


@dataclass(frozen=True)
class PhraseCorpus:
    path: Path
    source_sha256: str
    corpus_sha256: str
    policy_digest: str
    lineage_digest: str
    token_count: int
    joined_count: int


def materialize_phrase_corpus(
    source: Path, destination: Path, policy: PhrasePolicy
) -> PhraseCorpus:
    """Apply the original word2phrase score in stable left-to-right passes.

    Lines remain document boundaries. Tokens are arbitrary non-whitespace bytes and joined
    phrases therefore retain the engine's byte-token contract.
    """
    source = Path(source)
    if not source.is_file():
        raise ValueError(f"phrase source does not exist: {source}")
    source_sha = _sha256(source)
    policy_payload = policy.record()
    policy_digest = _digest(policy_payload)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    current = source
    temporary_paths: list[Path] = []
    joined_count = 0
    try:
        for pass_index in range(policy.passes):
            output = destination.with_name(
                f".{destination.name}.pass-{pass_index}-{os.getpid()}"
            )
            temporary_paths.append(output)
            joined_count += _phrase_pass(current, output, policy)
            current = output
        os.replace(current, destination)
        temporary_paths.remove(current)
    finally:
        for path in temporary_paths:
            path.unlink(missing_ok=True)
    corpus_sha = _sha256(destination)
    token_count = sum(
        len(line.split()) for line in destination.read_bytes().splitlines()
    )
    lineage = {
        "format": "f2-phrase-corpus-v1",
        "source_sha256": source_sha,
        "corpus_sha256": corpus_sha,
        "policy": policy_payload,
        "policy_digest": policy_digest,
        "token_count": token_count,
        "joined_count": joined_count,
    }
    lineage_digest = _digest(lineage)
    lineage["lineage_digest"] = lineage_digest
    sidecar = destination.with_suffix(destination.suffix + ".json")
    sidecar.write_text(json.dumps(lineage, indent=2, sort_keys=True) + "\n")
    return PhraseCorpus(
        destination,
        source_sha,
        corpus_sha,
        policy_digest,
        lineage_digest,
        token_count,
        joined_count,
    )


def _phrase_pass(source: Path, destination: Path, policy: PhrasePolicy) -> int:
    lines = [line.split() for line in source.read_bytes().splitlines()]
    unigrams = Counter(token for line in lines for token in line)
    bigrams = Counter(pair for line in lines for pair in pairwise(line))
    token_total = sum(unigrams.values())
    joined = 0
    with destination.open("wb") as stream:
        for tokens in lines:
            result: list[bytes] = []
            index = 0
            while index < len(tokens):
                if index + 1 < len(tokens):
                    left, right = tokens[index], tokens[index + 1]
                    count = bigrams[(left, right)]
                    score = (
                        (count - policy.min_count)
                        * token_total
                        / (unigrams[left] * unigrams[right])
                    )
                    if count >= policy.min_count and score > policy.threshold:
                        result.append(left + policy.separator + right)
                        joined += 1
                        index += 2
                        continue
                result.append(tokens[index])
                index += 1
            stream.write(b" ".join(result) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    return joined


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["PhraseCorpus", "PhrasePolicy", "materialize_phrase_corpus"]
