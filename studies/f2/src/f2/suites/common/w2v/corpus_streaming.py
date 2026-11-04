"""Corpus streaming and prefix slicing routines."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from repro_io.checksum import sha256_file

_RUNS = re.compile(rb"[ \t\r\n]+|[^ \t\r\n]+")
CORPUS_CHUNK_BYTES = 64 * 1024


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
            if b"\x00" in chunk:
                chunk = chunk.replace(b"\x00", b" ")
            if b"</s>" in chunk:
                chunk = chunk.replace(b"</s>", b" ")
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


def slice_corpus_file(
    source_path: Path,
    target_path: Path,
    budget: int,
    *,
    source_sha256: str | None = None,
    scratch_dir: Path | None = None,
) -> tuple[int, str]:
    """Slice exact `budget` tokens from source_path to target_path using _copy_corpus_prefix.

    Verifies input identity using source_sha256 if provided, or source file size and mtime,
    avoiding expensive re-hashing of large source corpora on cache hit.
    """
    meta_path = target_path.with_suffix(".json")
    if target_path.is_file() and meta_path.is_file() and source_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            cached_tokens = meta.get("lexical_tokens")
            source_stat = source_path.stat()
            source_matches = True
            if source_sha256 is not None:
                source_matches = meta.get("source_sha256") == source_sha256
            elif "source_size" in meta and "source_mtime_ns" in meta:
                source_matches = (
                    meta.get("source_size") == source_stat.st_size
                    and meta.get("source_mtime_ns") == source_stat.st_mtime_ns
                )

            if (
                source_matches
                and meta.get("lexical_token_budget") == budget
                and isinstance(cached_tokens, int)
                and 0 < cached_tokens <= budget
                and target_path.stat().st_size == meta.get("byte_size")
            ):
                return cached_tokens, str(meta["corpus_sha256"])
        except Exception:
            pass

    target_path.parent.mkdir(parents=True, exist_ok=True)
    scratch = scratch_dir or target_path.parent
    scratch.mkdir(parents=True, exist_ok=True)
    temp_target = target_path.with_suffix(f".tmp-{os.getpid()}")
    try:

        def file_chunks():
            with source_path.open("rb") as f:
                while True:
                    chunk = f.read(CORPUS_CHUNK_BYTES)
                    if not chunk:
                        break
                    yield chunk

        with temp_target.open("wb") as dest:
            tokens, _ = _copy_corpus_prefix(
                file_chunks(), dest, budget, scratch=scratch
            )
            dest.flush()
            os.fsync(dest.fileno())

        sha256 = sha256_file(temp_target)
        temp_target.replace(target_path)
        source_stat = source_path.stat()
        meta = {
            "source": str(source_path),
            "source_size": source_stat.st_size,
            "source_mtime_ns": source_stat.st_mtime_ns,
            "source_sha256": source_sha256,
            "lexical_token_budget": budget,
            "lexical_tokens": tokens,
            "corpus_sha256": sha256,
            "byte_size": target_path.stat().st_size,
        }
        meta_temp = meta_path.with_suffix(f".tmp-{os.getpid()}")
        meta_temp.write_text(
            json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        meta_temp.replace(meta_path)
        return tokens, sha256
    finally:
        temp_target.unlink(missing_ok=True)


__all__ = [
    "CORPUS_CHUNK_BYTES",
    "_copy_corpus_prefix",
    "_decompressed_chunks",
    "slice_corpus_file",
]
