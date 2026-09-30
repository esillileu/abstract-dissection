"""Feasibility analysis and yield estimation for FineWeb 2013 news surrogate."""

from __future__ import annotations

import math
import statistics
import time
from dataclasses import asdict, dataclass
from typing import Any

from ...corpus.canonical.normalization import normalize_text
from .adapter import (
    FINEWEB_DEFAULT_DUMP,
    FineWebSourceAdapter,
)
from .classifier import FineWebNewsClassifier


@dataclass
class FeasibilityReport:
    sample_size: int
    dump: str
    documents_seen: int
    source_words_seen: int
    news_documents_accepted: int
    accepted_w2v_words: int
    document_acceptance_ratio: float
    word_acceptance_ratio: float
    mean_source_doc_length: float
    mean_accepted_doc_length: float
    median_accepted_doc_length: float
    min_words_filter: int
    read_throughput_docs_sec: float
    filter_throughput_docs_sec: float
    prep_throughput_docs_sec: float
    total_throughput_docs_sec: float
    source_words_required_for_33b: int
    source_docs_required_for_33b: int
    dump_total_documents: int
    estimated_dump_total_words: int
    estimated_dump_accepted_words: int
    can_fulfill_33b_with_dump_20: bool
    requires_dump_48: bool
    estimated_storage_gb_33b: float
    margin_ratio_20: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_feasibility_study(
    *,
    sample_size: int = 1000,
    dump: str = FINEWEB_DEFAULT_DUMP,
    min_words: int = 100,
) -> FeasibilityReport:
    """Run an empirical feasibility analysis on a sample of FineWeb documents."""
    adapter = FineWebSourceAdapter()
    first_file = adapter.list_parquet_files(dump)[0]
    url = adapter.resolve_url(first_file)

    t0_read = time.perf_counter()
    docs = list(adapter.iter_remote_parquet(url, dump=dump, limit=sample_size))
    t1_read = time.perf_counter()
    read_duration = max(t1_read - t0_read, 1e-6)

    documents_seen = len(docs)
    source_doc_lengths: list[int] = []
    accepted_doc_lengths: list[int] = []
    total_source_words = 0
    total_accepted_words = 0
    accepted_docs_count = 0

    filter_duration = 0.0
    prep_duration = 0.0

    for doc in docs:
        raw_words = len(doc.text.split())
        source_doc_lengths.append(raw_words)
        total_source_words += raw_words

        if raw_words < min_words:
            continue

        t_f0 = time.perf_counter()
        score, is_news, _ = FineWebNewsClassifier.evaluate(doc.text, url=doc.url)
        t_f1 = time.perf_counter()
        filter_duration += t_f1 - t_f0

        if not is_news:
            continue

        t_p0 = time.perf_counter()
        normalized = normalize_text(doc.text)
        words = len(normalized.split())
        t_p1 = time.perf_counter()
        prep_duration += t_p1 - t_p0

        accepted_docs_count += 1
        total_accepted_words += words
        accepted_doc_lengths.append(words)

    total_duration = read_duration + filter_duration + prep_duration

    doc_ratio = accepted_docs_count / max(documents_seen, 1)
    word_ratio = total_accepted_words / max(total_source_words, 1)
    mean_source_len = statistics.mean(source_doc_lengths) if source_doc_lengths else 0.0
    mean_accepted_len = (
        statistics.mean(accepted_doc_lengths) if accepted_doc_lengths else 0.0
    )
    median_accepted_len = (
        float(statistics.median(accepted_doc_lengths)) if accepted_doc_lengths else 0.0
    )

    # 33B estimation
    target_33b = 33_000_000_000
    source_words_required = (
        int(math.ceil(target_33b / max(word_ratio, 1e-9))) if word_ratio > 0 else 0
    )
    source_docs_required = (
        int(math.ceil(source_words_required / max(mean_source_len, 1.0)))
        if mean_source_len > 0
        else 0
    )

    # CC-MAIN-2013-20 known capacity (from HuggingFace card)
    dump_20_docs = 215_280_647
    est_dump_words = int(dump_20_docs * mean_source_len)
    est_accepted_words = int(est_dump_words * word_ratio)

    can_fulfill = est_accepted_words >= target_33b
    margin = est_accepted_words / target_33b if target_33b > 0 else 0.0

    # Storage estimation: ~5.5 bytes per raw W2V word uncompressed, ~2.0 bytes zstd -19 compressed
    est_storage_gb = (target_33b * 2.0) / (1024**3)

    return FeasibilityReport(
        sample_size=sample_size,
        dump=dump,
        documents_seen=documents_seen,
        source_words_seen=total_source_words,
        news_documents_accepted=accepted_docs_count,
        accepted_w2v_words=total_accepted_words,
        document_acceptance_ratio=doc_ratio,
        word_acceptance_ratio=word_ratio,
        mean_source_doc_length=mean_source_len,
        mean_accepted_doc_length=mean_accepted_len,
        median_accepted_doc_length=median_accepted_len,
        min_words_filter=min_words,
        read_throughput_docs_sec=documents_seen / max(read_duration, 1e-6),
        filter_throughput_docs_sec=documents_seen / max(filter_duration, 1e-6),
        prep_throughput_docs_sec=(
            accepted_docs_count / max(prep_duration, 1e-6)
            if accepted_docs_count > 0
            else 0.0
        ),
        total_throughput_docs_sec=documents_seen / max(total_duration, 1e-6),
        source_words_required_for_33b=source_words_required,
        source_docs_required_for_33b=source_docs_required,
        dump_total_documents=dump_20_docs,
        estimated_dump_total_words=est_dump_words,
        estimated_dump_accepted_words=est_accepted_words,
        can_fulfill_33b_with_dump_20=can_fulfill,
        requires_dump_48=not can_fulfill,
        estimated_storage_gb_33b=est_storage_gb,
        margin_ratio_20=margin,
    )


__all__ = [
    "FeasibilityReport",
    "run_feasibility_study",
]
