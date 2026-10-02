"""Filtering, W2V normalization, and word-counting pipeline for FineWeb."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

from ...corpus.canonical.normalization import normalize_text
from .adapter import FINEWEB_DATASET, FINEWEB_PINNED_REVISION
from .classifier import FineWebNewsClassifier
from .models import FineWebDocument, FineWebNewsRecord, FineWebProvenanceRecord


@dataclass
class PipelineStatistics:
    """Cumulative metrics tracked across the FineWeb filtering pipeline."""

    documents_seen: int = 0
    documents_accepted: int = 0
    total_raw_words: int = 0
    total_accepted_words: int = 0
    too_short_rejected: int = 0
    non_news_rejected: int = 0
    scores: list[float] = field(default_factory=list)


class FineWebPipeline:
    """Applies journalistic news classification and Word2Vec normalization."""

    def __init__(
        self,
        *,
        min_words: int = 100,
        news_threshold: float = 1.5,
        dataset: str = FINEWEB_DATASET,
        revision: str = FINEWEB_PINNED_REVISION,
    ) -> None:
        self.min_words = min_words
        self.news_threshold = news_threshold
        self.dataset = dataset
        self.revision = revision
        self.stats = PipelineStatistics()

    def process_document(
        self, doc: FineWebDocument
    ) -> tuple[FineWebNewsRecord | None, FineWebProvenanceRecord | None]:
        """Evaluate one document; if accepted as news, return normalized and provenance records."""
        self.stats.documents_seen += 1
        raw_words = len(doc.text.split())
        self.stats.total_raw_words += raw_words

        if raw_words < self.min_words:
            self.stats.too_short_rejected += 1
            return None, None

        score, is_news, _ = FineWebNewsClassifier.evaluate(doc.text, url=doc.url)
        self.stats.scores.append(score)

        if not is_news or score < self.news_threshold:
            self.stats.non_news_rejected += 1
            return None, None

        # Apply W2V C-locale normalization
        normalized = normalize_text(doc.text)
        word_count = len(normalized.split())

        self.stats.documents_accepted += 1
        self.stats.total_accepted_words += word_count

        news_record = FineWebNewsRecord(
            canonical_id=doc.canonical_id,
            normalized_text=normalized,
            word_count=word_count,
            news_score=score,
            doc=doc,
        )

        stem = doc.parquet_path.rsplit("/", 1)[-1]
        provenance_record = FineWebProvenanceRecord(
            canonical_id=doc.canonical_id,
            fineweb_id=doc.fineweb_id,
            dataset=self.dataset,
            revision=self.revision,
            dump=doc.dump,
            parquet_file=stem,
            row_index=doc.row_index,
            url=doc.url,
            date=doc.date,
            word_count=word_count,
            news_score=score,
        )

        return news_record, provenance_record

    def process_stream(
        self, docs: Iterator[FineWebDocument]
    ) -> Iterator[tuple[FineWebNewsRecord, FineWebProvenanceRecord]]:
        """Filter and normalize a stream of documents."""
        for doc in docs:
            news_rec, prov_rec = self.process_document(doc)
            if news_rec is not None and prov_rec is not None:
                yield news_rec, prov_rec


__all__ = [
    "FineWebPipeline",
    "PipelineStatistics",
]
