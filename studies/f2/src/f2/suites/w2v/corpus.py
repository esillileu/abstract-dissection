"""Materialize immutable ordered corpus bindings for the Word2Vec engine."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from repro_io.checksum import sha256_file

from repro_core.context.paths import RuntimePaths

_RUNS = re.compile(rb"[ \t\r\n]+|[^ \t\r\n]+")
CORPUS_CHUNK_BYTES = 64 * 1024


class ObjectReader(Protocol):
    """The read-only part of ``repro_io.s3.S3ObjectStore`` used here."""

    def get_file(self, uri: str, local_path: Path) -> Path: ...


@dataclass(frozen=True)
class CorpusShard:
    index: int
    uri: str
    sha256: str
    byte_size: int
    word_count: int
    document_count: int


@dataclass(frozen=True)
class CorpusBinding:
    shards: tuple[CorpusShard, ...]

    @classmethod
    def from_rows(
        cls,
        rows: Sequence[Mapping[str, object]],
    ) -> CorpusBinding:
        """Translate verified corpus DB rows into ordered shard inputs."""
        return cls(
            tuple(
                CorpusShard(
                    index=int(row["shard_index"]),
                    uri=str(row["s3_uri"]),
                    sha256=str(row["sha256"]),
                    byte_size=int(row["byte_size"]),
                    word_count=int(row["word_count"]),
                    document_count=int(row["doc_count"]),
                )
                for row in rows
            )
        )


@dataclass(frozen=True)
class MaterializedCorpus:
    path: Path
    corpus_sha256: str
    lexical_tokens: int
    complete_shards: int


def _binding_manifest_key(binding: CorpusBinding, budget: int) -> str:
    payload = {
        "budget": budget,
        "format": "f2-w2v-materialized-v2",
        "shards": [
            (shard.index, shard.sha256, shard.byte_size) for shard in binding.shards
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class CorpusMaterializer:
    """Fetch, verify, cache, and stream an ordered binding to one local file."""

    def __init__(
        self,
        store: ObjectReader,
        *,
        paths: RuntimePaths | None = None,
        download_attempts: int = 2,
    ) -> None:
        if download_attempts < 1:
            raise ValueError("download_attempts must be positive")
        self.store = store
        self.paths = paths or RuntimePaths.from_environment()
        self.download_attempts = download_attempts

    def _corpus_root(self) -> Path:
        if hasattr(self.paths, "inputs"):
            root = self.paths.inputs("f2", "prepared", "corpus")
        else:
            root = self.paths.inputs_root / "f2" / "prepared" / "corpus"
        root.mkdir(parents=True, exist_ok=True)
        return root

    def materialize(
        self,
        binding: CorpusBinding,
        *,
        lexical_token_budget: int,
        progress: Callable[[int, int, int], None] | None = None,
    ) -> MaterializedCorpus:
        root = self._corpus_root()
        key = _binding_manifest_key(binding, lexical_token_budget)
        with (root / f"{key}.lock").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            return self._materialize_binding(
                binding, lexical_token_budget=lexical_token_budget, progress=progress
            )

    def _materialize_binding(self, binding, *, lexical_token_budget, progress):
        if lexical_token_budget < 1:
            raise ValueError("lexical_token_budget must be positive")
        self._validate_binding(binding)

        root = self._corpus_root()
        manifest_root = root / "manifests"
        manifest_root.mkdir(parents=True, exist_ok=True)
        manifest_key = _binding_manifest_key(binding, lexical_token_budget)
        manifest_file = manifest_root / f"{manifest_key}.json"

        cached = None
        if manifest_file.is_file():
            try:
                manifest_data = json.loads(manifest_file.read_text(encoding="utf-8"))
                if (
                    not isinstance(manifest_data, dict)
                    or manifest_data.get("manifest_key") != manifest_key
                ):
                    raise ValueError("corpus cache input identity mismatch")
                cached_sha256 = manifest_data["corpus_sha256"]
                cached_tokens = int(manifest_data["lexical_tokens"])
                cached_shards = int(manifest_data["complete_shards"])
                cached_size = int(manifest_data["byte_size"])
                final = root / "materialized" / f"{cached_sha256}.txt"
                if (
                    0 < cached_tokens <= lexical_token_budget
                    and 0 <= cached_shards <= len(binding.shards)
                    and final.is_file()
                    and final.stat().st_size == cached_size
                    and sha256_file(final) == cached_sha256
                ):
                    cached = MaterializedCorpus(
                        final,
                        cached_sha256,
                        cached_tokens,
                        cached_shards,
                    )
            except (ValueError, KeyError, TypeError, OSError):
                manifest_file.unlink(missing_ok=True)
        if cached is not None:
            if progress is not None:
                progress(
                    cached.complete_shards, len(binding.shards), cached.lexical_tokens
                )
            return cached

        work_root = root / "work"
        work_root.mkdir(parents=True, exist_ok=True)
        for abandoned in work_root.glob(f"{manifest_key}-*"):
            shutil.rmtree(abandoned)
        staging = Path(tempfile.mkdtemp(prefix=f"{manifest_key}-", dir=work_root))
        output = staging / "corpus.txt"
        try:
            tokens, complete_shards = self._write_corpus(
                binding,
                root / "objects",
                output,
                lexical_token_budget,
                progress=progress,
            )
            corpus_sha256 = sha256_file(output)
            final = root / "materialized" / f"{corpus_sha256}.txt"
            final.parent.mkdir(parents=True, exist_ok=True)
            if final.is_file() and sha256_file(final) == corpus_sha256:
                output.unlink()
            else:
                output.replace(final)
            _sync_directory(final.parent)

            manifest_data = {
                "manifest_key": manifest_key,
                "corpus_sha256": corpus_sha256,
                "lexical_tokens": tokens,
                "complete_shards": complete_shards,
                "byte_size": final.stat().st_size,
            }
            staging_manifest = staging / "manifest.json"
            staging_manifest.write_text(
                json.dumps(manifest_data, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with staging_manifest.open("rb") as stream:
                os.fsync(stream.fileno())
            os.replace(staging_manifest, manifest_file)
            _sync_directory(manifest_root)

            return MaterializedCorpus(
                final,
                corpus_sha256,
                tokens,
                complete_shards,
            )
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    @staticmethod
    def _validate_binding(binding: CorpusBinding) -> None:
        if not binding.shards:
            raise ValueError("corpus binding must contain shards")
        expected_indices = list(range(len(binding.shards)))
        actual_indices = [shard.index for shard in binding.shards]
        if actual_indices != expected_indices:
            raise ValueError("corpus shards must be ordered with contiguous indices")
        for shard in binding.shards:
            if (
                not shard.uri.startswith("s3://")
                or len(shard.sha256) != 64
                or shard.byte_size < 1
                or shard.word_count < 0
                or shard.document_count < 0
            ):
                raise ValueError(
                    f"invalid corpus shard metadata at index {shard.index}"
                )

    def _cached_shard(self, shard: CorpusShard, object_root: Path) -> Path:
        target = object_root / f"shard-{shard.index:05d}-{shard.sha256}.txt.zst"
        if self._valid_object(target, shard):
            return target
        object_root.mkdir(parents=True, exist_ok=True)
        for attempt in range(self.download_attempts):
            with tempfile.TemporaryDirectory(
                prefix=".download-", dir=object_root
            ) as work:
                temporary = Path(work) / "shard.txt.zst"
                try:
                    self.store.get_file(shard.uri, temporary)
                    if self._valid_object(temporary, shard):
                        os.replace(temporary, target)
                        return target
                except (OSError, RuntimeError):
                    if attempt + 1 == self.download_attempts:
                        raise
        raise OSError(
            f"downloaded corpus shard failed verification at index {shard.index}"
        )

    @staticmethod
    def _valid_object(path: Path, shard: CorpusShard) -> bool:
        return (
            path.is_file()
            and path.stat().st_size == shard.byte_size
            and sha256_file(path) == shard.sha256
        )

    def _write_corpus(
        self,
        binding: CorpusBinding,
        object_root: Path,
        output: Path,
        budget: int,
        *,
        progress: Callable[[int, int, int], None] | None,
    ) -> tuple[int, int]:
        tokens = 0
        complete_shards = 0
        with output.open("wb") as destination:
            for shard in binding.shards:
                source = self._cached_shard(shard, object_root)
                last_byte = None
                with closing(_decompressed_chunks(source)) as chunks:

                    def observed_chunks():
                        nonlocal last_byte
                        for chunk in chunks:
                            if chunk:
                                last_byte = chunk[-1:]
                            yield chunk

                    consumed, shard_complete = _copy_corpus_prefix(
                        observed_chunks(),
                        destination,
                        budget - tokens,
                        scratch=output.parent,
                    )
                tokens += consumed
                if shard_complete:
                    complete_shards += 1
                    if (
                        last_byte is not None
                        and last_byte != b"\n"
                        and tokens < budget
                        and shard.index + 1 < len(binding.shards)
                    ):
                        # A shard boundary must never concatenate two lexical tokens.
                        destination.write(b"\n")
                if progress is not None:
                    progress(shard.index + 1, len(binding.shards), tokens)
                if tokens == budget or not shard_complete:
                    break
            destination.flush()
            os.fsync(destination.fileno())
        if tokens == 0:
            raise ValueError("corpus binding contains no lexical tokens")
        return tokens, complete_shards


def _sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _copy_corpus_prefix(chunks, destination, budget, *, scratch):
    """Preserve the historical exact line/token cutoff with bounded RAM.

    Whitespace after the last selected token is spooled until we know whether
    its line ends (preserve it) or another token follows (discard it).
    """
    tokens = 0
    in_token = False
    line_has_selected = False
    with tempfile.SpooledTemporaryFile(
        max_size=CORPUS_CHUNK_BYTES, dir=scratch
    ) as pending:

        def flush_whitespace():
            pending.seek(0)
            shutil.copyfileobj(pending, destination, length=CORPUS_CHUNK_BYTES)
            pending.seek(0)
            pending.truncate()

        for chunk in chunks:
            for match in _RUNS.finditer(chunk):
                part = match.group()
                if part[0] not in b" \t\r\n":
                    if not in_token:
                        if tokens == budget:
                            if line_has_selected:
                                destination.write(b"\n")
                            return tokens, False
                        tokens += 1
                        line_has_selected = True
                    destination.write(part)
                    in_token = True
                else:
                    in_token = False
                    if tokens < budget:
                        destination.write(part)
                        if b"\n" in part:
                            line_has_selected = False
                    else:
                        pieces = part.split(b"\n")
                        for piece in pieces[:-1]:
                            pending.write(piece + b"\n")
                            flush_whitespace()
                            line_has_selected = False
                        pending.write(pieces[-1])
        flush_whitespace()
    return tokens, True


def _decompressed_chunks(path: Path):
    process = subprocess.Popen(
        ["zstd", "-dc", "-q", path.as_posix()], stdout=subprocess.PIPE
    )
    assert process.stdout is not None
    try:
        while chunk := process.stdout.read(CORPUS_CHUNK_BYTES):
            yield chunk
        if process.wait():
            raise RuntimeError(
                f"zstd decompression failed for cached shard {path.name}"
            )
    finally:
        process.stdout.close()
        if process.poll() is None:
            process.terminate()
        process.wait()


__all__ = [
    "CorpusBinding",
    "CorpusMaterializer",
    "CorpusShard",
    "MaterializedCorpus",
]
