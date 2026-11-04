"""Deterministic phrase-corpus construction and lineage."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from w2v import phrase_pass as _native_phrase_pass

from .phrase_policy import (
    TOKEN_CHUNK_BYTES,
    PhrasePolicy,
    _tokens,
    _validate_token,
)


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
    source_sha256: str | None = None,
    _verified_source_sha256: str | None = None,
) -> PhraseCorpus:
    """Apply the historical F2 token-total-normalized score in stable left-to-right passes."""
    source = Path(source)
    if not source.is_file():
        raise ValueError(f"phrase source does not exist: {source}")
    source_sha = _verified_source_sha256 or _sha256(source)
    if source_sha256 is not None and source_sha != source_sha256:
        raise ValueError("phrase source checksum mismatch")
    policy_payload = policy.record()
    policy_digest = _digest(policy_payload)
    destination = Path(destination)
    if destination.resolve() == source.resolve():
        raise ValueError("phrase destination must differ from source")
    destination.parent.mkdir(parents=True, exist_ok=True)
    current = source
    temporary_paths: list[Path] = []
    joined_count = 0
    pass_statistics: list[dict[str, object]] = []
    try:
        for pass_index in range(policy.passes):
            if progress is not None:
                progress(f"phrase pass={pass_index + 1}/{policy.passes} counting")
            output = destination.with_name(
                f".{destination.name}.pass-{pass_index}-{os.getpid()}"
            )
            temporary_paths.append(output)
            stats = _phrase_pass(
                current,
                output,
                policy,
                pass_index=pass_index + 1,
                pass_count=policy.passes,
                progress=progress,
            )
            pass_statistics.append(stats)
            joined_count += int(stats["joined_count"])
            if current != source:
                current.unlink()
                temporary_paths.remove(current)
            current = output
        os.replace(current, destination)
        temporary_paths.remove(current)
    finally:
        for path in temporary_paths:
            path.unlink(missing_ok=True)
    if _sha256(source) != source_sha:
        destination.unlink(missing_ok=True)
        raise ValueError("phrase source changed during materialization")
    corpus_sha = str(pass_statistics[-1]["corpus_sha256"])
    token_count = int(pass_statistics[-1]["output_tokens"])
    lineage = {
        "format": "f2-phrase-corpus-v2",
        "source_sha256": source_sha,
        "corpus_sha256": corpus_sha,
        "policy": policy_payload,
        "policy_digest": policy_digest,
        "token_count": token_count,
        "joined_count": joined_count,
        "pass_statistics": pass_statistics,
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
    threads: int = 0,
) -> dict[str, object]:
    threshold = (policy.thresholds or (policy.threshold,) * policy.passes)[
        pass_index - 1
    ]
    with tempfile.TemporaryDirectory(
        prefix="phrase-counts-", dir=destination.parent
    ) as work:
        (
            token_total,
            document_count,
            unique_unigrams,
            unique_bigrams,
            joined,
            output_tokens,
        ) = _native_phrase_pass(
            source,
            destination,
            Path(work),
            policy.min_count,
            threshold,
            policy.separator,
            pass_index=pass_index,
            pass_count=pass_count,
            chunk_bytes=TOKEN_CHUNK_BYTES,
            threads=threads,
            progress=progress,
        )
    with destination.open("rb") as stream:
        os.fsync(stream.fileno())
    if progress is not None:
        progress(
            f"phrase pass={pass_index}/{pass_count} complete "
            f"documents={document_count} joined={joined}"
        )
    return {
        "pass": pass_index,
        "threshold": threshold,
        "input_tokens": token_total,
        "documents": document_count,
        "unique_unigrams": unique_unigrams,
        "unique_bigrams": unique_bigrams,
        "joined_count": joined,
        "output_tokens": output_tokens,
        "corpus_sha256": _sha256(destination),
    }


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


def _phrase_root(cache_root: Path) -> Path:
    p = Path(cache_root)
    if p.name == "phrases":
        return p
    if p.name == "prepared" or (p / "phrases").is_dir():
        return p / "phrases"
    return p / "f2/w2v/phrases"


def load_cached_phrase_corpus(
    cache_root: Path, source_sha256: str, policy: PhrasePolicy
) -> PhraseCorpus:
    key = _digest({"source_sha256": source_sha256, "policy": policy.record()})
    path = _phrase_root(cache_root) / key / "phrases.txt"
    lineage = json.loads(path.with_suffix(".txt.json").read_text())
    if not isinstance(lineage, dict):
        raise ValueError("invalid phrase lineage")
    digest = lineage.pop("lineage_digest")
    if (
        digest != _digest(lineage)
        or lineage["source_sha256"] != source_sha256
        or lineage["policy"] != policy.record()
        or lineage["corpus_sha256"] != _sha256(path)
    ):
        raise ValueError("invalid phrase cache")
    return PhraseCorpus(
        path,
        source_sha256,
        lineage["corpus_sha256"],
        lineage["policy_digest"],
        digest,
        lineage["token_count"],
        lineage["joined_count"],
    )


def cached_phrase_corpus(
    source: Path,
    cache_root: Path,
    policy: PhrasePolicy,
    *,
    progress=None,
    source_sha256=None,
) -> PhraseCorpus:
    """Serialize builders and publish a complete, verified immutable directory."""
    source = Path(source)
    root = _phrase_root(cache_root)
    root.mkdir(parents=True, exist_ok=True)

    if source_sha256 is not None:
        key = _digest({"source_sha256": source_sha256, "policy": policy.record()})
        entry = root / key
        with (root / f"{key}.lock").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            for abandoned in root.glob(f".{key}-*"):
                shutil.rmtree(abandoned)
            if entry.exists():
                try:
                    return load_cached_phrase_corpus(cache_root, source_sha256, policy)
                except (ValueError, KeyError, OSError, TypeError):
                    shutil.rmtree(entry)

    source_sha = _sha256(source)
    if source_sha256 is not None and source_sha256 != source_sha:
        raise ValueError("phrase source checksum mismatch")
    key = _digest({"source_sha256": source_sha, "policy": policy.record()})
    entry = root / key
    with (root / f"{key}.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for abandoned in root.glob(f".{key}-*"):
            shutil.rmtree(abandoned)
        if entry.exists():
            try:
                return load_cached_phrase_corpus(cache_root, source_sha, policy)
            except (ValueError, KeyError, OSError, TypeError):
                shutil.rmtree(entry)
        with tempfile.TemporaryDirectory(prefix=f".{key}-", dir=root) as work:
            result = materialize_phrase_corpus(
                source,
                Path(work) / "phrases.txt",
                policy,
                progress=progress,
                source_sha256=source_sha,
                _verified_source_sha256=source_sha,
            )
            with result.path.with_suffix(".txt.json").open("rb") as stream:
                os.fsync(stream.fileno())
            descriptor = os.open(work, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            os.rename(work, entry)
            descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        return PhraseCorpus(
            entry / "phrases.txt",
            result.source_sha256,
            result.corpus_sha256,
            result.policy_digest,
            result.lineage_digest,
            result.token_count,
            result.joined_count,
        )


__all__ = [
    "TOKEN_CHUNK_BYTES",
    "PhraseCorpus",
    "PhrasePolicy",
    "_tokens",
    "_validate_token",
    "cached_phrase_corpus",
    "materialize_phrase_corpus",
]
