"""Materialize immutable ordered corpus bindings for the Word2Vec engine."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
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
    resource_version_id: str
    manifest_digest: str
    shards: tuple[CorpusShard, ...]

    @classmethod
    def from_rows(
        cls,
        resource_version_id: str,
        manifest_digest: str,
        rows: Sequence[Mapping[str, object]],
    ) -> CorpusBinding:
        """Translate catalog/corpus repository rows into an immutable binding."""
        return cls(
            resource_version_id=resource_version_id,
            manifest_digest=manifest_digest,
            shards=tuple(
                CorpusShard(
                    index=int(row["shard_index"]),
                    uri=str(row["s3_uri"]),
                    sha256=str(row["sha256"]),
                    byte_size=int(row["byte_size"]),
                    word_count=int(row["word_count"]),
                    document_count=int(row["doc_count"]),
                )
                for row in rows
            ),
        )


@dataclass(frozen=True)
class MaterializedCorpus:
    path: Path
    resource_version_id: str
    manifest_digest: str
    corpus_sha256: str
    lexical_tokens: int
    complete_shards: int


def ordered_manifest_digest(shards: Sequence[CorpusShard]) -> str:
    """Hash the ordered identity fields recorded by the F2 corpus catalog."""
    payload = [asdict(shard) for shard in shards]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
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

    def materialize(
        self, binding: CorpusBinding, *, lexical_token_budget: int
    ) -> MaterializedCorpus:
        if lexical_token_budget < 1:
            raise ValueError("lexical_token_budget must be positive")
        self._validate_binding(binding)

        root = (
            self.paths.cache_root
            / "exp"
            / "f2"
            / "w2v"
            / "corpus"
            / binding.resource_version_id
            / binding.manifest_digest
        )
        root.mkdir(parents=True, exist_ok=True)
        final = root / f"tokens-{lexical_token_budget}.txt"
        metadata = final.with_suffix(".json")
        cached = self._read_result(metadata, final, binding, lexical_token_budget)
        if cached is not None:
            return cached

        staging = (
            self.paths.staging_root
            / "exp"
            / "f2"
            / "w2v"
            / "corpus"
            / f"{binding.manifest_digest}-{lexical_token_budget}-{os.getpid()}"
        )
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        output = staging / "corpus.txt"
        try:
            tokens, complete_shards = self._write_corpus(
                binding, root / "objects", output, lexical_token_budget
            )
            corpus_sha256 = sha256_file(output)
            result_payload = {
                "schema_version": 1,
                "resource_version_id": binding.resource_version_id,
                "manifest_digest": binding.manifest_digest,
                "lexical_token_budget": lexical_token_budget,
                "lexical_tokens": tokens,
                "complete_shards": complete_shards,
                "corpus_sha256": corpus_sha256,
            }
            output.replace(final)
            temporary_metadata = staging / "materialization.json"
            temporary_metadata.write_text(
                json.dumps(result_payload, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            temporary_metadata.replace(metadata)
            return MaterializedCorpus(
                final,
                binding.resource_version_id,
                binding.manifest_digest,
                corpus_sha256,
                tokens,
                complete_shards,
            )
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    @staticmethod
    def _validate_binding(binding: CorpusBinding) -> None:
        if not binding.resource_version_id or not binding.shards:
            raise ValueError("corpus binding must identify a version and shards")
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
        if ordered_manifest_digest(binding.shards) != binding.manifest_digest:
            raise ValueError("ordered corpus manifest digest mismatch")

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
                if tokens == budget or not shard_complete:
                    break
            destination.flush()
            os.fsync(destination.fileno())
        if tokens == 0:
            raise ValueError("corpus binding contains no lexical tokens")
        return tokens, complete_shards

    @staticmethod
    def _read_result(
        metadata: Path,
        corpus: Path,
        binding: CorpusBinding,
        budget: int,
    ) -> MaterializedCorpus | None:
        if not metadata.is_file() or not corpus.is_file():
            return None
        try:
            payload = json.loads(metadata.read_text(encoding="utf-8"))
            digest = sha256_file(corpus)
            if (
                payload["schema_version"] != 1
                or payload["resource_version_id"] != binding.resource_version_id
                or payload["manifest_digest"] != binding.manifest_digest
                or payload["lexical_token_budget"] != budget
                or payload["corpus_sha256"] != digest
            ):
                return None
            return MaterializedCorpus(
                corpus,
                binding.resource_version_id,
                binding.manifest_digest,
                digest,
                int(payload["lexical_tokens"]),
                int(payload["complete_shards"]),
            )
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            return None


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
    "ordered_manifest_digest",
]
