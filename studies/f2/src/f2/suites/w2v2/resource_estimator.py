"""Resource estimation and profiling algorithms for large-scale W2V2 training."""

from __future__ import annotations

import math
import shutil
import sqlite3
import tempfile
from collections import Counter
from pathlib import Path

from .sentences import sentences


def disk_free(path: Path) -> int:
    while not path.exists():
        path = path.parent
    return shutil.disk_usage(path).free


def profile_sentences(lengths: dict[int, int]) -> dict:
    count = sum(lengths.values())
    tokens = sum(n * frequency for n, frequency in lengths.items())
    pairs = sum(n * (n - 1) * frequency for n, frequency in lengths.items())

    def percentile(fraction: float) -> int:
        position = max(1, math.ceil(count * fraction))
        seen = 0
        for length, frequency in sorted(lengths.items()):
            seen += frequency
            if seen >= position:
                return length
        return 0

    largest = sorted(lengths.items(), reverse=True)
    tail_count = max(1, math.ceil(count * 0.01))
    tail_pairs = 0
    for n, frequency in largest:
        take = min(frequency, tail_count)
        tail_pairs += take * n * (n - 1)
        tail_count -= take
        if tail_count == 0:
            break
    return dict(
        sentences=count,
        lexical_tokens=tokens,
        max_sentence_tokens=max(lengths, default=0),
        p50=percentile(0.5),
        p95=percentile(0.95),
        p99=percentile(0.99),
        ordered_pairs=pairs,
        longest_one_percent_pair_fraction=tail_pairs / pairs if pairs else 0,
        length_histogram={
            str(n): frequency for n, frequency in sorted(lengths.items())
        },
    )


def measure_phrase(path: Path, vocabulary_config: dict, work: Path) -> dict:
    """Count exact types with bounded SQLite memory before allocating a vocabulary."""
    work.mkdir(parents=True, exist_ok=True)
    for abandoned in work.glob("vocabulary-counts-*"):
        shutil.rmtree(abandoned)
    with tempfile.TemporaryDirectory(
        prefix="vocabulary-counts-", dir=work
    ) as temporary:
        with sqlite3.connect(Path(temporary) / "counts.sqlite") as db:
            db.execute("PRAGMA cache_size=-16384")
            db.execute("PRAGMA journal_mode=OFF")
            db.execute(
                "CREATE TABLE counts (token BLOB PRIMARY KEY, frequency INTEGER) WITHOUT ROWID"
            )
            lengths: Counter[int] = Counter()
            batch: Counter[bytes] = Counter()
            for sentence in sentences(path):
                lengths[len(sentence)] += 1
                batch.update(sentence)
                if len(batch) >= 1_000_000:
                    db.executemany(
                        "INSERT INTO counts VALUES (?, ?) ON CONFLICT(token) DO UPDATE SET frequency = frequency + excluded.frequency",
                        batch.items(),
                    )
                    batch.clear()
            if batch:
                db.executemany(
                    "INSERT INTO counts VALUES (?, ?) ON CONFLICT(token) DO UPDATE SET frequency = frequency + excluded.frequency",
                    batch.items(),
                )
            db.commit()
            min_count = vocabulary_config.get("min_count", 5)
            row = db.execute(
                "SELECT COUNT(*), COALESCE(SUM(frequency), 0), SUM(CASE WHEN frequency >= ? THEN 1 ELSE 0 END), SUM(CASE WHEN frequency >= ? THEN frequency ELSE 0 END) FROM counts",
                (min_count, min_count),
            ).fetchone()
            total_types, total_tokens, retained_types, retained_tokens = row
            return dict(
                raw_token_count=total_tokens or 0,
                unique_token_types=total_types or 0,
                retained_token_count=retained_tokens or 0,
                retained_token_types=retained_types or 0,
                sentence_profile=profile_sentences(lengths),
            )


def resource_estimate(
    config: dict,
    *,
    measured_tokens: int,
    measured_retained_types: int,
    source_bytes: int,
) -> dict:
    """Predict physical requirements from verified inputs without training."""
    training = config["training"]
    dimension = int(training["embedding_dimension"])
    thread_count = int(training["thread_count"])
    hash_capacity = int(config.get("vocabulary", {}).get("hash_capacity", 30_000_000))
    unigram_entries = int(training.get("negative_table_size", 0))
    table_bytes = unigram_entries * 4

    input_matrix = measured_retained_types * dimension * 4
    output_matrix = measured_retained_types * dimension * 4
    hash_bytes = hash_capacity * 4
    counts_bytes = hash_capacity * 8
    huffman_bytes = hash_capacity * 32
    token_bytes = measured_retained_types * 40
    vocabulary_bytes = (
        hash_bytes + counts_bytes + huffman_bytes + token_bytes + table_bytes
    )
    worker_bytes = thread_count * ((1 << 20) + (1000 * 4) + (256 * 1024))
    rss = (
        input_matrix
        + output_matrix
        + vocabulary_bytes
        + worker_bytes
        + (256 * 1024 * 1024)
    )

    checkpoint = (
        input_matrix
        + output_matrix
        + (measured_retained_types * 20)
        + (16 * 1024 * 1024)
    )
    lookup = input_matrix + (measured_retained_types * 20)
    best_retained = 2 * checkpoint
    staging = (2 * checkpoint) + best_retained + lookup

    return dict(
        predicted_peak_rss_bytes=rss,
        matrix_bytes=input_matrix + output_matrix,
        vocabulary_bytes=vocabulary_bytes,
        single_checkpoint_bytes=checkpoint,
        lookup_artifact_bytes=lookup,
        additional_staging_disk_bytes=staging,
    )


__all__ = [
    "disk_free",
    "measure_phrase",
    "profile_sentences",
    "resource_estimate",
]
