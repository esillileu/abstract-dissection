"""Deterministic phrase-corpus construction and lineage."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import sqlite3
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

# Full-token training preserves phrase identity. Memory is bounded by fixed I/O
# batches plus the largest individual token, never the largest document.
TOKEN_CHUNK_BYTES = 64 * 1024
_WHITESPACE = re.compile(rb"([ \t\r\n\v\f]+)")


def _validate_token(token: bytes) -> None:
    if b"\0" in token or token == b"</s>":
        raise ValueError(
            "phrase token violates engine contract (no NUL or reserved </s>)"
        )


def _tokens(path: Path):
    """Yield byte tokens and LF boundaries with bounded chunk/token storage."""
    pending = bytearray()
    unfinished_line = False
    with path.open("rb") as stream:
        while chunk := stream.read(TOKEN_CHUNK_BYTES):
            unfinished_line = (
                bool(chunk.rsplit(b"\n", 1)[-1]) if b"\n" in chunk else True
            )
            for index, part in enumerate(_WHITESPACE.split(chunk)):
                if index % 2 == 0:
                    pending.extend(part)
                else:
                    if pending:
                        _validate_token(pending)
                        yield bytes(pending)
                        pending.clear()
                    for _ in range(part.count(b"\n")):
                        yield None
        if pending:
            _validate_token(pending)
            yield bytes(pending)
        if unfinished_line:
            yield None


@dataclass(frozen=True)
class PhrasePolicy:
    passes: int
    threshold: float
    min_count: int
    separator: bytes = b"_"
    thresholds: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if (
            self.passes < 1
            or not math.isfinite(self.threshold)
            or self.threshold < 0
            or self.min_count < 1
        ):
            raise ValueError("invalid phrase detection policy")
        if self.thresholds is not None and (
            len(self.thresholds) != self.passes
            or any(not math.isfinite(t) or t < 0 for t in self.thresholds)
        ):
            raise ValueError(
                "thresholds must contain one finite nonnegative value per pass"
            )
        if (
            not self.separator
            or b"\0" in self.separator
            or any(byte in b" \t\r\n\v\f" for byte in self.separator)
        ):
            raise ValueError("phrase separator must be non-empty non-whitespace bytes")

    @classmethod
    def from_config(cls, values: Mapping[str, object]) -> PhrasePolicy:
        thresholds = values.get("thresholds")
        if thresholds is not None and not isinstance(thresholds, (list, tuple)):
            raise ValueError("phrase thresholds must be a list")
        return cls(
            passes=int(values["passes"]),
            threshold=float(values.get("threshold", 100.0)),
            min_count=int(values["min_count"]),
            separator=str(values.get("separator", "_")).encode("ascii"),
            thresholds=None
            if thresholds is None
            else tuple(float(t) for t in thresholds),
        )

    def record(self) -> dict[str, object]:
        return {
            "score_normalization": "token_total",
            "token_contract": "full_byte_tokens_no_nul_no_eos",
            "thresholds": list(self.thresholds or (self.threshold,) * self.passes),
            "passes": self.passes,
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
    source_sha256: str | None = None,
) -> PhraseCorpus:
    """Apply the historical F2 token-total-normalized score in stable left-to-right passes.

    Lines remain document boundaries. Tokens are arbitrary non-whitespace bytes and joined
    phrases therefore retain the engine's byte-token contract.
    """
    source = Path(source)
    if not source.is_file():
        raise ValueError(f"phrase source does not exist: {source}")
    source_sha = _sha256(source)
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
    corpus_sha = _sha256(destination)
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
) -> dict[str, object]:
    # A private scratch database has a fixed 16 MiB page cache. Counts are exact;
    # neither vocabulary nor bigram cardinality determines resident memory.
    with tempfile.TemporaryDirectory(
        prefix="phrase-counts-", dir=destination.parent
    ) as work:
        with sqlite3.connect(Path(work) / "counts.sqlite") as db:
            db.execute("PRAGMA cache_size=-16384")
            db.execute("PRAGMA temp_store=FILE")
            db.execute("PRAGMA journal_mode=OFF")
            db.execute(
                "CREATE TABLE unigram (token BLOB PRIMARY KEY, n INTEGER) WITHOUT ROWID"
            )
            db.execute(
                "CREATE TABLE bigram (a BLOB, b BLOB, n INTEGER, PRIMARY KEY(a,b)) WITHOUT ROWID"
            )
            token_total = 0
            document_count = 0
            unigrams: list[tuple[bytes]] = []
            bigrams: list[tuple[bytes, bytes]] = []
            previous = None
            batch_bytes = 0

            def flush_counts():
                db.executemany(
                    "INSERT INTO unigram VALUES (?,1) ON CONFLICT(token) DO UPDATE SET n=n+1",
                    unigrams,
                )
                db.executemany(
                    "INSERT INTO bigram VALUES (?,?,1) ON CONFLICT(a,b) DO UPDATE SET n=n+1",
                    bigrams,
                )
                db.commit()
                unigrams.clear()
                bigrams.clear()

            for token in _tokens(source):
                if token is None:
                    previous = None
                    document_count += 1
                    if progress is not None and document_count % 100_000 == 0:
                        progress(
                            f"phrase pass={pass_index}/{pass_count} counted_documents={document_count}"
                        )
                    continue
                token_total += 1
                unigrams.append((token,))
                batch_bytes += len(token)
                if previous is not None:
                    bigrams.append((previous, token))
                previous = token
                if len(unigrams) >= 8192 or batch_bytes >= 1024 * 1024:
                    flush_counts()
                    batch_bytes = 0
            flush_counts()
            threshold = (policy.thresholds or (policy.threshold,) * policy.passes)[
                pass_index - 1
            ]
            # Preserve Python's historical floating-point operation order.
            db.create_function(
                "score",
                3,
                lambda n, a, b: (n - policy.min_count) * token_total / (a * b),
            )
            db.execute(
                "CREATE TABLE eligible (a BLOB, b BLOB, PRIMARY KEY(a,b)) WITHOUT ROWID"
            )
            db.execute(
                "INSERT INTO eligible SELECT b.a,b.b FROM bigram b "
                "JOIN unigram l ON l.token=b.a JOIN unigram r ON r.token=b.b "
                "WHERE b.n>=? AND score(b.n,l.n,r.n)>?",
                (policy.min_count, threshold),
            )
            stats = {
                "pass": pass_index,
                "threshold": threshold,
                "input_tokens": token_total,
                "documents": document_count,
                "unique_unigrams": db.execute(
                    "SELECT count(*) FROM unigram"
                ).fetchone()[0],
                "unique_bigrams": db.execute("SELECT count(*) FROM bigram").fetchone()[
                    0
                ],
            }
            joined = _write_pass(
                source,
                destination,
                policy,
                db,
                pass_index,
                pass_count,
                document_count,
                progress,
            )
            return stats | {
                "joined_count": joined,
                "output_tokens": token_total - joined,
                "corpus_sha256": _sha256(destination),
            }


def _write_pass(
    source, destination, policy, db, pass_index, pass_count, document_count, progress
):
    joined = 0
    left = None
    first = True
    written_documents = 0
    with destination.open("wb") as stream:

        def emit(token):
            nonlocal first
            if not first:
                stream.write(b" ")
            stream.write(token)
            first = False

        for token in _tokens(source):
            if token is None:
                if left is not None:
                    emit(left)
                    left = None
                stream.write(b"\n")
                first = True
                written_documents += 1
                if progress is not None and written_documents % 100_000 == 0:
                    progress(
                        f"phrase pass={pass_index}/{pass_count} written_documents={written_documents}/{document_count}"
                    )
            elif left is None:
                left = token
            elif (
                db.execute(
                    "SELECT 1 FROM eligible WHERE a=? AND b=?", (left, token)
                ).fetchone()
                is not None
            ):
                phrase = left + policy.separator + token
                _validate_token(phrase)
                emit(phrase)
                joined += 1
                left = None
            else:
                emit(left)
                left = token
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
    source_sha = _sha256(source)
    if source_sha256 is not None and source_sha256 != source_sha:
        raise ValueError("phrase source checksum mismatch")
    root = Path(cache_root) / "f2/w2v/phrases"
    root.mkdir(parents=True, exist_ok=True)
    key = _digest({"source_sha256": source_sha, "policy": policy.record()})
    entry = root / key
    with (root / f"{key}.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        # A killed writer can leave only private work directories, never an entry.
        for abandoned in root.glob(f".{key}-*"):
            shutil.rmtree(abandoned)
        if entry.exists():
            try:
                path = entry / "phrases.txt"
                lineage = json.loads(path.with_suffix(".txt.json").read_text())
                if not isinstance(lineage, dict):
                    raise ValueError("invalid phrase lineage")
                digest = lineage.pop("lineage_digest")
                if (
                    digest != _digest(lineage)
                    or lineage["source_sha256"] != source_sha
                    or lineage["policy"] != policy.record()
                    or lineage["corpus_sha256"] != _sha256(path)
                ):
                    raise ValueError("invalid phrase cache")
                return PhraseCorpus(
                    path,
                    source_sha,
                    lineage["corpus_sha256"],
                    lineage["policy_digest"],
                    digest,
                    lineage["token_count"],
                    lineage["joined_count"],
                )
            except (ValueError, KeyError, OSError, TypeError):
                shutil.rmtree(entry)
        with tempfile.TemporaryDirectory(prefix=f".{key}-", dir=root) as work:
            result = materialize_phrase_corpus(
                source,
                Path(work) / "phrases.txt",
                policy,
                progress=progress,
                source_sha256=source_sha,
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
    "PhraseCorpus",
    "PhrasePolicy",
    "cached_phrase_corpus",
    "materialize_phrase_corpus",
]
