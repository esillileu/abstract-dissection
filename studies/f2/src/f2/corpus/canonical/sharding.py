"""Deterministic zstd-compressed text sharding and streaming."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

from repro_io.checksum import sha256_file

CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True)
class ShardInfo:
    index: int
    path: str
    physical_sha256: str
    logical_sha256: str
    compressed_bytes: int
    uncompressed_bytes: int
    word_count: int
    record_count: int
    source_span: dict[str, object]
    newline_count: int = 0
    train_words_count: int = 0


class DeterministicSharder:
    def __init__(self, output_dir: Path, target_words: int = 10_000_000) -> None:
        self.output_dir = output_dir
        self.target_words = target_words

    def write(
        self,
        records: Iterable[tuple[str, str]],
        *,
        source: str,
        boundary_policy: str = "sentence_per_line",
    ) -> list[ShardInfo]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        shards: list[ShardInfo] = []
        batch: list[tuple[str, str]] = []
        words = 0

        def flush() -> None:
            nonlocal batch, words
            if not batch:
                return
            index = len(shards)
            raw = b"".join(
                text.rstrip("\n").encode("utf-8") + b"\n" for _, text in batch
            )
            newline_count = raw.count(b"\n")
            train_words_count = words + newline_count
            raw_path = self.output_dir / f"shard-{index:05d}.txt"
            compressed_path = raw_path.with_suffix(".txt.zst")
            raw_path.write_bytes(raw)
            env = os.environ.copy()
            env["LC_ALL"] = "C"
            subprocess.run(
                [
                    "zstd",
                    "-q",
                    "-f",
                    "-T1",
                    "-19",
                    "--no-progress",
                    raw_path.as_posix(),
                    "-o",
                    compressed_path.as_posix(),
                ],
                check=True,
                env=env,
            )
            raw_path.unlink()
            shards.append(
                ShardInfo(
                    index,
                    compressed_path.name,
                    sha256_file(compressed_path),
                    hashlib.sha256(raw).hexdigest(),
                    compressed_path.stat().st_size,
                    len(raw),
                    words,
                    len(batch),
                    {"source": source, "first": batch[0][0], "last": batch[-1][0]},
                    newline_count=newline_count,
                    train_words_count=train_words_count,
                )
            )
            batch, words = [], 0

        for record_id, text in records:
            count = len(text.split())
            if batch and words + count > self.target_words:
                flush()
            batch.append((record_id, text))
            words += count
            if count > self.target_words:
                flush()
        flush()
        total_words = sum(shard.word_count for shard in shards)
        total_newlines = sum(shard.newline_count for shard in shards)
        manifest = {
            "schema_version": 2,
            "source": source,
            "target_words": self.target_words,
            "boundary_policy": boundary_policy,
            "summary": {
                "total_shards": len(shards),
                "total_lexical_words": total_words,
                "total_newlines": total_newlines,
                "total_word2vec_train_words": total_words + total_newlines,
            },
            "shards": [asdict(shard) for shard in shards],
        }
        (self.output_dir / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        return shards


def iter_shard_text(shard_path: Path) -> Iterator[str]:
    """Stream decompressed lines from a .txt.zst shard file using zstd CLI."""
    env = os.environ.copy()
    env["LC_ALL"] = "C"
    proc = subprocess.Popen(
        ["zstd", "-dc", "-q", shard_path.as_posix()],
        stdout=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    assert proc.stdout is not None
    yield from proc.stdout
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"zstd decompression failed for {shard_path}")


def open_canonical_shards(shard_paths: list[Path]) -> Iterator[tuple[str, str]]:
    """Read (record_id, text) from a list of source-canonical shard files."""
    for shard_path in sorted(shard_paths, key=lambda p: p.name):
        for lineno, line in enumerate(iter_shard_text(shard_path)):
            line = line.strip()
            if line:
                yield f"{shard_path.name}:{lineno}", line


__all__ = [
    "CHUNK_SIZE",
    "DeterministicSharder",
    "ShardInfo",
    "iter_shard_text",
    "open_canonical_shards",
]
