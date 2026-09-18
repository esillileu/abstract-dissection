"""End-to-end extraction and filtering pipeline orchestrator."""

from __future__ import annotations

from repro_io.archive import ARCParser

from .classifier import NewsClassifier
from .extractor import TextExtractor
from .filters import LanguageFilter, ValidityFilter, WordCounter
from .models import ProcessedDocumentResult


class PipelineRunner:
    """Executes the logical end-to-end extraction and filtering pipeline."""

    def __init__(self, min_words: int = 100) -> None:
        self.min_words = min_words

    def process(
        self,
        record_id: str,
        crawl_id: str,
        url: str,
        raw_arc_compressed: bytes,
        inclusion_probability: float,
        design_weight: float,
        downloaded_bytes: int,
    ) -> ProcessedDocumentResult:
        if not raw_arc_compressed:
            return ProcessedDocumentResult(
                record_id=record_id,
                crawl_id=crawl_id,
                url=url,
                fetch_status="fetch_failed",
                downloaded_bytes=downloaded_bytes,
                http_status=0,
                extraction_success=False,
                clean_text=None,
                news_score=0.0,
                is_news_predicted=False,
                is_english=False,
                is_valid=False,
                rejection_reason="no_data",
                word_count=0,
                inclusion_probability=inclusion_probability,
                design_weight=design_weight,
                proxy_words=0,
                diagnostics={},
            )

        # 1. Parse ARC
        arc_record = ARCParser.parse_arc_bytes(raw_arc_compressed)
        if arc_record is None or arc_record.http_status != 200:
            return ProcessedDocumentResult(
                record_id=record_id,
                crawl_id=crawl_id,
                url=url,
                fetch_status="arc_parse_failed",
                downloaded_bytes=downloaded_bytes,
                http_status=arc_record.http_status if arc_record else 0,
                extraction_success=False,
                clean_text=None,
                news_score=0.0,
                is_news_predicted=False,
                is_english=False,
                is_valid=False,
                rejection_reason="arc_parse_failed",
                word_count=0,
                inclusion_probability=inclusion_probability,
                design_weight=design_weight,
                proxy_words=0,
                diagnostics={},
            )

        # 2. Extract Text
        clean_text = TextExtractor.extract_text(
            arc_record.html_body, url=arc_record.url or url
        )
        if not clean_text:
            return ProcessedDocumentResult(
                record_id=record_id,
                crawl_id=crawl_id,
                url=url,
                fetch_status="extraction_failed",
                downloaded_bytes=downloaded_bytes,
                http_status=200,
                extraction_success=False,
                clean_text=None,
                news_score=0.0,
                is_news_predicted=False,
                is_english=False,
                is_valid=False,
                rejection_reason="no_article_text",
                word_count=0,
                inclusion_probability=inclusion_probability,
                design_weight=design_weight,
                proxy_words=0,
                diagnostics={},
            )

        # 3. News Classification
        news_score, is_news, news_details = NewsClassifier.evaluate(
            clean_text, arc_record.html_body, url
        )

        # 4. Language Filter
        is_en, lang_conf, detected_lang = LanguageFilter.is_english(clean_text)

        # 5. Validity Filter
        is_valid, reject_reason = ValidityFilter.check_validity(
            clean_text, min_words=self.min_words
        )

        # 6. Word Count
        words = WordCounter.count_words(clean_text)
        proxy_words = words if (is_news and is_en and is_valid) else 0

        return ProcessedDocumentResult(
            record_id=record_id,
            crawl_id=crawl_id,
            url=url,
            fetch_status="success",
            downloaded_bytes=downloaded_bytes,
            http_status=200,
            extraction_success=True,
            clean_text=clean_text,
            news_score=news_score,
            is_news_predicted=is_news,
            is_english=is_en,
            is_valid=is_valid,
            rejection_reason=reject_reason,
            word_count=words,
            inclusion_probability=inclusion_probability,
            design_weight=design_weight,
            proxy_words=proxy_words,
            diagnostics={
                "detected_lang": detected_lang,
                "lang_conf": lang_conf,
                "news_details": news_details,
            },
        )


__all__ = ["PipelineRunner"]
