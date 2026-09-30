"""Interruption-safe production materializer for the FineWeb 2013 33B surrogate corpus."""

from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from repro_io.http.download import BandwidthScheduler, SerialDownloader

from ...corpus.canonical.sharding import ShardInfo
from .adapter import (
    FINEWEB_DEFAULT_DUMP,
    FineWebSourceAdapter,
)
from .pipeline import FineWebPipeline
from .writer import FineWebShardWriter


@dataclass
class MaterializerCheckpoint:
    spec_hash: str
    target_words: int
    cumulative_words: int = 0
    documents_seen: int = 0
    documents_accepted: int = 0
    last_completed_file_index: int = -1
    completed_shards: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MaterializerCheckpoint:
        return cls(**data)


class ParquetPrefetcher:
    """Downloads Parquet files in the background ahead of the reader pipeline."""

    def __init__(
        self,
        downloader: SerialDownloader,
        adapter: FineWebSourceAdapter,
        parquet_files: list[str],
        staging_dir: Path,
        start_index: int,
        dump: str,
        max_buffered: int = 1,
    ) -> None:
        self.downloader = downloader
        self.adapter = adapter
        self.parquet_files = parquet_files
        self.staging_dir = staging_dir
        self.start_index = start_index
        self.dump = dump
        self.queue: queue.Queue[tuple[int, Path] | None] = queue.Queue(
            maxsize=max_buffered
        )
        self.worker_exc: Exception | None = None
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="FineWebPrefetcher"
        )
        self._thread.start()

    def _run(self) -> None:
        for file_idx in range(self.start_index, len(self.parquet_files)):
            if self._stop_event.is_set():
                break
            rel_path = self.parquet_files[file_idx]
            url = self.adapter.resolve_url(rel_path)
            local_parquet = self.staging_dir / f"download_{file_idx:05d}.parquet"

            import pyarrow.parquet as pq

            already_valid = False
            if local_parquet.is_file():
                try:
                    pq.ParquetFile(str(local_parquet))
                    already_valid = True
                except Exception:
                    local_parquet.unlink(missing_ok=True)

            try:
                if not already_valid:
                    self.downloader.download(url, local_parquet)
                while not self._stop_event.is_set():
                    try:
                        self.queue.put((file_idx, local_parquet), timeout=0.5)
                        break
                    except queue.Full:
                        continue
            except Exception as e:
                self.worker_exc = e
                break

        while not self._stop_event.is_set():
            try:
                self.queue.put(None, timeout=0.5)
                break
            except queue.Full:
                continue

    def stop(self) -> None:
        self._stop_event.set()


class FineWebProductionMaterializer:
    """Manages downloading, rate-limiting, pipeline execution, and checkpointing."""

    def __init__(
        self,
        output_dir: Path,
        *,
        target_words: int = 33_000_000_000,
        target_words_per_shard: int = 10_000_000,
        peak_mbps: float = 40.0,
        offpeak_mbps: float = 100.0,
        no_bandwidth_limit: bool = False,
        min_words: int = 100,
        dump: str = FINEWEB_DEFAULT_DUMP,
    ) -> None:
        self.output_dir = output_dir
        self.target_words = target_words
        self.target_words_per_shard = target_words_per_shard
        self.dump = dump
        self.min_words = min_words

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.staging_dir = self.output_dir / ".staging"
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = self.output_dir / "checkpoint.json"

        # Initialize rate-limited downloader
        bw: BandwidthScheduler | None = None
        if not no_bandwidth_limit:
            bw = BandwidthScheduler(peak_mbps=peak_mbps, offpeak_mbps=offpeak_mbps)
        self.downloader = SerialDownloader(
            bandwidth=bw,
            retries=8,
            backoff_seconds=2.0,
            timeout=120.0,
        )
        self.adapter = FineWebSourceAdapter()

    def load_checkpoint(self) -> MaterializerCheckpoint:
        if self.checkpoint_path.is_file():
            data = json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
            return MaterializerCheckpoint.from_dict(data)
        from .spec import PRODUCTION_SPEC

        return MaterializerCheckpoint(
            spec_hash=PRODUCTION_SPEC.spec_hash,
            target_words=self.target_words,
        )

    def save_checkpoint(self, cp: MaterializerCheckpoint) -> None:
        temp_path = self.checkpoint_path.with_suffix(".tmp")
        temp_path.write_text(
            json.dumps(cp.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temp_path.replace(self.checkpoint_path)

    def run(self, *, max_parquet_files: int | None = None) -> dict[str, Any]:
        checkpoint = self.load_checkpoint()
        parquet_files = self.adapter.list_parquet_files(self.dump)
        if max_parquet_files is not None:
            parquet_files = parquet_files[:max_parquet_files]

        writer = FineWebShardWriter(
            self.output_dir,
            target_words=self.target_words_per_shard,
            dump=self.dump,
        )
        # Restore already recorded shards into writer
        restored = [ShardInfo(**s_dict) for s_dict in checkpoint.completed_shards]
        writer.load_completed_shards(restored)

        pipeline = FineWebPipeline(min_words=self.min_words)
        pipeline.stats.documents_seen = checkpoint.documents_seen
        pipeline.stats.documents_accepted = checkpoint.documents_accepted
        pipeline.stats.total_accepted_words = checkpoint.cumulative_words

        t0_start = time.time()
        start_index = checkpoint.last_completed_file_index + 1

        import sys

        from tqdm import tqdm

        is_tty = sys.stderr.isatty()
        pbar = tqdm(
            total=self.target_words,
            initial=pipeline.stats.total_accepted_words,
            desc="FineWeb 33B",
            unit="words",
            unit_scale=True,
            dynamic_ncols=True,
            mininterval=1.0 if is_tty else 10.0,
        )

        prefetcher = ParquetPrefetcher(
            downloader=self.downloader,
            adapter=self.adapter,
            parquet_files=parquet_files,
            staging_dir=self.staging_dir,
            start_index=start_index,
            dump=self.dump,
        )

        try:
            while True:
                if pipeline.stats.total_accepted_words >= self.target_words:
                    break

                if prefetcher.worker_exc is not None:
                    raise prefetcher.worker_exc

                item = prefetcher.queue.get()
                if item is None:
                    if prefetcher.worker_exc is not None:
                        raise prefetcher.worker_exc
                    break
                file_idx, local_parquet = item

                try:
                    # 1. Stream documents sequentially from the local Parquet file
                    doc_stream = self.adapter.iter_local_parquet(
                        local_parquet, dump=self.dump
                    )
                    for news_rec, prov_rec in pipeline.process_stream(doc_stream):
                        writer.append(news_rec, prov_rec)
                        pbar.update(news_rec.word_count)
                        if pipeline.stats.total_accepted_words >= self.target_words:
                            break
                finally:
                    # 2. Clean up the downloaded raw Parquet file immediately
                    local_parquet.unlink(missing_ok=True)
                    prefetcher.queue.task_done()

                # 3. Wait for all pending shard compressions from this file
                writer.sync()
                pbar.set_postfix(
                    shards=len(writer.shards),
                    file=f"{file_idx + 1}/{len(parquet_files)}",
                )

                # 4. Save checkpoint after each Parquet file
                checkpoint.cumulative_words = pipeline.stats.total_accepted_words
                checkpoint.documents_seen = pipeline.stats.documents_seen
                checkpoint.documents_accepted = pipeline.stats.documents_accepted
                checkpoint.last_completed_file_index = file_idx
                checkpoint.completed_shards = [asdict(s) for s in writer.shards]
                self.save_checkpoint(checkpoint)
        finally:
            prefetcher.stop()
            pbar.close()

        # Finalize any remaining documents in writer batch
        shards = writer.close()
        checkpoint.completed_shards = [asdict(s) for s in shards]
        self.save_checkpoint(checkpoint)

        elapsed = time.time() - t0_start
        return {
            "target_words": self.target_words,
            "cumulative_words": pipeline.stats.total_accepted_words,
            "documents_seen": pipeline.stats.documents_seen,
            "documents_accepted": pipeline.stats.documents_accepted,
            "shards_count": len(shards),
            "files_processed": checkpoint.last_completed_file_index + 1,
            "elapsed_seconds": elapsed,
            "manifest_path": str(writer.manifest_path),
            "provenance_path": str(writer.provenance_path),
            "checkpoint_path": str(self.checkpoint_path),
        }


__all__ = [
    "FineWebProductionMaterializer",
    "MaterializerCheckpoint",
]
