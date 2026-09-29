"""Materialize immutable ordered corpus bindings for the Word2Vec engine."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from repro_io.checksum import sha256_file

from repro_core.context.paths import RuntimePaths

_TOKEN = re.compile(rb"[^\x20\x09\x0a\x0d]+")


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

    def materialize(
        self,
        binding: CorpusBinding,
        *,
        lexical_token_budget: int,
        progress: Callable[[int, int, int], None] | None = None,
    ) -> MaterializedCorpus:
        if lexical_token_budget < 1:
            raise ValueError("lexical_token_budget must be positive")
        self._validate_binding(binding)

        root = self.paths.cache_root / "exp" / "f2" / "w2v" / "corpus"
        root.mkdir(parents=True, exist_ok=True)

        staging = (
            self.paths.staging_root
            / "exp"
            / "f2"
            / "w2v"
            / "corpus"
            / f"materialize-{lexical_token_budget}-{os.getpid()}"
        )
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
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
        target.unlink(missing_ok=True)
        for attempt in range(self.download_attempts):
            try:
                self.store.get_file(shard.uri, target)
                if self._valid_object(target, shard):
                    return target
            except (OSError, RuntimeError):
                if attempt + 1 == self.download_attempts:
                    raise
            target.unlink(missing_ok=True)
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
                shard_complete = True
                for line in _decompressed_lines(source):
                    matches = list(_TOKEN.finditer(line))
                    available = budget - tokens
                    if len(matches) <= available:
                        destination.write(line)
                        tokens += len(matches)
                        continue
                    if available:
                        destination.write(
                            line[: matches[available - 1].end()].rstrip(b"\r\n")
                        )
                        destination.write(b"\n")
                        tokens += available
                    shard_complete = False
                    break
                if shard_complete:
                    complete_shards += 1
                if progress is not None:
                    progress(shard.index + 1, len(binding.shards), tokens)
                if tokens == budget or not shard_complete:
                    break
            destination.flush()
            os.fsync(destination.fileno())
        if tokens == 0:
            raise ValueError("corpus binding contains no lexical tokens")
        return tokens, complete_shards


def _decompressed_lines(path: Path):
    process = subprocess.Popen(
        ["zstd", "-dc", "-q", path.as_posix()], stdout=subprocess.PIPE
    )
    assert process.stdout is not None
    try:
        yield from process.stdout
    finally:
        process.stdout.close()
        return_code = process.wait()
    if return_code:
        raise RuntimeError(f"zstd decompression failed for cached shard {path.name}")


__all__ = [
    "CorpusBinding",
    "CorpusMaterializer",
    "CorpusShard",
    "MaterializedCorpus",
]
