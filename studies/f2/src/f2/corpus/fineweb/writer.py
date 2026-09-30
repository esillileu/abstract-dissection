"""Deterministic shard and provenance writer for FineWeb surrogate corpus."""

from __future__ import annotations

import hashlib
import json
import os
import queue
import subprocess
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from repro_io.checksum import sha256_file

from ...corpus.canonical.sharding import ShardInfo
from .adapter import FINEWEB_DATASET, FINEWEB_PINNED_REVISION
from .models import FineWebNewsRecord, FineWebProvenanceRecord


@dataclass
class _CompressionTask:
    index: int
    raw: bytes
    word_count: int
    record_count: int
    first_id: str
    last_id: str
    newline_count: int
    train_words_count: int


class FineWebShardWriter:
    """Writes compressed zstd shards asynchronously via a bounded queue, provenance records, and manifest.json."""

    def __init__(
        self,
        output_dir: Path,
        *,
        target_words: int = 10_000_000,
        dataset: str = FINEWEB_DATASET,
        revision: str = FINEWEB_PINNED_REVISION,
        dump: str = "CC-MAIN-2013-20",
        queue_maxsize: int = 2,
    ) -> None:
        self.output_dir = output_dir
        self.target_words = target_words
        self.dataset = dataset
        self.revision = revision
        self.dump = dump

        self.shards_dir = output_dir / "shards"
        self.shards_dir.mkdir(parents=True, exist_ok=True)
        self.provenance_path = output_dir / "provenance.jsonl"
        self.manifest_path = output_dir / "manifest.json"

        self.shards: list[ShardInfo] = []
        self._current_batch: list[FineWebNewsRecord] = []
        self._current_words: int = 0
        self._prov_file = self.provenance_path.open("a", encoding="utf-8")

        self._next_shard_index: int = 0
        self._lock = threading.Lock()
        self._queue: queue.Queue[_CompressionTask | None] = queue.Queue(
            maxsize=queue_maxsize
        )
        self._worker_exc: Exception | None = None
        self._worker = threading.Thread(
            target=self._compress_worker, daemon=True, name="FineWebCompressor"
        )
        self._worker.start()

    def load_completed_shards(self, shards: list[ShardInfo]) -> None:
        """Initialize writer with already recorded shards from checkpoint."""
        with self._lock:
            self.shards = list(shards)
            self._next_shard_index = len(self.shards)

    def _compress_worker(self) -> None:
        env = os.environ.copy()
        env["LC_ALL"] = "C"
        while True:
            task = self._queue.get()
            if task is None:
                self._queue.task_done()
                break
            try:
                raw_path = self.shards_dir / f"shard-{task.index:05d}.txt"
                compressed_path = raw_path.with_suffix(".txt.zst")
                raw_path.write_bytes(task.raw)
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

                shard_info = ShardInfo(
                    index=task.index,
                    path=f"shards/{compressed_path.name}",
                    physical_sha256=sha256_file(compressed_path),
                    logical_sha256=hashlib.sha256(task.raw).hexdigest(),
                    compressed_bytes=compressed_path.stat().st_size,
                    uncompressed_bytes=len(task.raw),
                    word_count=task.word_count,
                    record_count=task.record_count,
                    source_span={
                        "dataset": self.dataset,
                        "revision": self.revision,
                        "dump": self.dump,
                        "first": task.first_id,
                        "last": task.last_id,
                    },
                    newline_count=task.newline_count,
                    train_words_count=task.train_words_count,
                )
                with self._lock:
                    self.shards.append(shard_info)
            except Exception as e:
                self._worker_exc = e
            finally:
                self._queue.task_done()

    def append(
        self, news_rec: FineWebNewsRecord, prov_rec: FineWebProvenanceRecord
    ) -> None:
        """Add one accepted document, flushing to a new shard when word threshold is reached."""
        if self._worker_exc is not None:
            raise self._worker_exc

        self._prov_file.write(json.dumps(prov_rec.to_dict(), sort_keys=True) + "\n")
        self._prov_file.flush()

        count = news_rec.word_count
        if self._current_batch and self._current_words + count > self.target_words:
            self._flush()

        self._current_batch.append(news_rec)
        self._current_words += count

        if count >= self.target_words:
            self._flush()

    def sync(self) -> None:
        """Flush current batch and wait until all pending compression tasks are done."""
        self._flush()
        self._queue.join()
        if self._worker_exc is not None:
            raise self._worker_exc

    def close(self) -> list[ShardInfo]:
        """Finalize all remaining records and write the manifest."""
        self._flush()
        self._queue.put(None)
        self._worker.join()
        if self._worker_exc is not None:
            raise self._worker_exc

        self._prov_file.close()
        self._write_manifest()
        return self.shards

    def _flush(self) -> None:
        if self._worker_exc is not None:
            raise self._worker_exc

        if not self._current_batch:
            return

        with self._lock:
            if self._next_shard_index < len(self.shards):
                self._next_shard_index = len(self.shards)
            index = self._next_shard_index
            self._next_shard_index += 1

        raw = b"".join(
            rec.normalized_text.rstrip("\n").encode("utf-8") + b"\n"
            for rec in self._current_batch
        )
        newline_count = raw.count(b"\n")
        train_words_count = self._current_words + newline_count

        task = _CompressionTask(
            index=index,
            raw=raw,
            word_count=self._current_words,
            record_count=len(self._current_batch),
            first_id=self._current_batch[0].canonical_id,
            last_id=self._current_batch[-1].canonical_id,
            newline_count=newline_count,
            train_words_count=train_words_count,
        )
        self._current_batch = []
        self._current_words = 0

        # Blocks if queue has maxsize items -> Backpressure!
        self._queue.put(task)

    def _write_manifest(self) -> None:
        total_lexical = sum(s.word_count for s in self.shards)
        total_newlines = sum(s.newline_count for s in self.shards)
        manifest: dict[str, Any] = {
            "schema_version": 2,
            "corpus_identity": {
                "name": "fineweb-2013-news-surrogate",
                "dataset": self.dataset,
                "revision": self.revision,
                "dump": self.dump,
                "recipe": "mikolov_demo_train_big_model_v1_normalize_text",
                "news_filter": "FineWebNewsClassifier-v1",
                "surrogate_notice": (
                    "This corpus is a FineWeb 2013 news-domain surrogate for Word2Vec training, "
                    "not an exact reconstruction of Google News."
                ),
            },
            "target_words_per_shard": self.target_words,
            "summary": {
                "total_shards": len(self.shards),
                "total_records": sum(s.record_count for s in self.shards),
                "total_lexical_words": total_lexical,
                "total_newlines": total_newlines,
                "total_word2vec_train_words": total_lexical + total_newlines,
            },
            "shards": [asdict(s) for s in self.shards],
        }
        self.manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


__all__ = ["FineWebShardWriter"]
