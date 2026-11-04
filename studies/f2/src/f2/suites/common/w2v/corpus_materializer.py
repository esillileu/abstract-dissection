"""Corpus binding execution and cache materialization manager."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Callable
from contextlib import closing
from pathlib import Path

from repro_io.checksum import sha256_file

from repro_core.context.paths import RuntimePaths

from .corpus_streaming import _copy_corpus_prefix, _decompressed_chunks
from .corpus_types import (
    CorpusBinding,
    CorpusShard,
    MaterializedCorpus,
    ObjectReader,
)


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


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


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

    def _validate_binding(self, binding: CorpusBinding) -> None:
        if not binding.shards:
            raise ValueError("corpus binding must contain at least one shard")
        indices = [shard.index for shard in binding.shards]
        if indices != list(range(len(binding.shards))):
            raise ValueError("corpus binding shards must be dense and 0-indexed")

    def _write_corpus(
        self,
        binding: CorpusBinding,
        objects_root: Path,
        destination: Path,
        budget: int,
        *,
        progress: Callable[[int, int, int], None] | None = None,
    ) -> tuple[int, int]:
        objects_root.mkdir(parents=True, exist_ok=True)
        tokens = 0
        complete_shards = 0
        destination.parent.mkdir(parents=True, exist_ok=True)
        scratch = destination.parent / "scratch"
        scratch.mkdir(parents=True, exist_ok=True)
        try:
            with destination.open("wb") as output:
                for shard in binding.shards:
                    local_shard = self._acquire_shard(shard, objects_root)
                    with closing(_decompressed_chunks(local_shard)) as chunks:
                        shard_tokens, shard_complete = _copy_corpus_prefix(
                            chunks, output, budget - tokens, scratch=scratch
                        )
                    tokens += shard_tokens
                    if shard_complete:
                        complete_shards += 1
                    if progress is not None:
                        progress(complete_shards, len(binding.shards), tokens)
                    if tokens == budget:
                        break
                output.flush()
                os.fsync(output.fileno())
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
        if tokens == 0:
            raise ValueError("corpus binding contains no lexical tokens")
        return tokens, complete_shards

    def _acquire_shard(self, shard: CorpusShard, objects_root: Path) -> Path:
        target = objects_root / f"{shard.sha256}.zst"
        if target.is_file() and sha256_file(target) == shard.sha256:
            return target
        work = target.with_name(f".{target.name}.tmp-{os.getpid()}")
        work.unlink(missing_ok=True)
        last_error = None
        for _ in range(self.download_attempts):
            try:
                self.store.get_file(shard.uri, work)
                if sha256_file(work) != shard.sha256:
                    raise ValueError(f"shard checksum mismatch: {shard.uri}")
                os.replace(work, target)
                return target
            except Exception as exc:
                last_error = exc
                work.unlink(missing_ok=True)
        raise RuntimeError(
            f"failed to acquire shard {shard.uri} after {self.download_attempts} attempts"
        ) from last_error


__all__ = [
    "CorpusMaterializer",
]
