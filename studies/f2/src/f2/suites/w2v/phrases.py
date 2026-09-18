"""Deterministic phrase-corpus construction and lineage."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from collections.abc import Callable
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
    source: Path,
    destination: Path,
    policy: PhrasePolicy,
    *,
    progress: Callable[[str], None] | None = None,
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
            if progress is not None:
                progress(f"phrase pass={pass_index + 1}/{policy.passes} counting")
            output = destination.with_name(
                f".{destination.name}.pass-{pass_index}-{os.getpid()}"
            )
            temporary_paths.append(output)
            joined_count += _phrase_pass(
                current,
                output,
                policy,
                pass_index=pass_index + 1,
                pass_count=policy.passes,
                progress=progress,
            )
            current = output
        os.replace(current, destination)
        temporary_paths.remove(current)
    finally:
        for path in temporary_paths:
            path.unlink(missing_ok=True)
    corpus_sha = _sha256(destination)
    token_count = 0
    with destination.open("rb") as stream:
        for line in stream:
            token_count += len(line.split())
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


def _phrase_pass(
    source: Path,
    destination: Path,
    policy: PhrasePolicy,
    *,
    pass_index: int,
    pass_count: int,
    progress: Callable[[str], None] | None,
) -> int:
    unigrams: Counter[bytes] = Counter()
    bigrams: Counter[tuple[bytes, bytes]] = Counter()
    document_count = 0
    with source.open("rb") as source_stream:
        for document_count, line in enumerate(source_stream, start=1):
            tokens = line.split()
            unigrams.update(tokens)
            bigrams.update(pairwise(tokens))
            if progress is not None and document_count % 100_000 == 0:
                progress(
                    f"phrase pass={pass_index}/{pass_count} "
                    f"counted_documents={document_count}"
                )
    token_total = sum(unigrams.values())
    joined = 0
    with source.open("rb") as source_stream, destination.open("wb") as stream:
        for written_documents, line in enumerate(source_stream, start=1):
            tokens = line.split()
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
            if progress is not None and written_documents % 100_000 == 0:
                progress(
                    f"phrase pass={pass_index}/{pass_count} "
                    f"written_documents={written_documents}/{document_count}"
                )
        stream.flush()
        os.fsync(stream.fileno())
    if progress is not None:
        progress(
            f"phrase pass={pass_index}/{pass_count} complete "
            f"documents={document_count} joined={joined}"
        )
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
