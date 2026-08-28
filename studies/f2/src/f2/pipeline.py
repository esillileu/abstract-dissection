"""Pure processing pipeline: ARC parsing (warcio), Trafilatura extraction, explicit news classification, language & validity filters, and word counting."""

from __future__ import annotations

import gzip
import io
import re
from dataclasses import dataclass
from typing import Any

import langdetect
import trafilatura
from warcio.archiveiterator import ArchiveIterator

# Compile standard regexes for content classification and validation
DATELINE_PATTERNS = [
    re.compile(
        r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2},?\s+\d{4}\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
    re.compile(r"\b\d{1,2}/\d{1,2}/\d{4}\b"),
]

BYLINE_PATTERNS = [
    re.compile(
        r"\b(?:by|written by|author:?)\s+[A-Z][a-z]+\s+[A-Z][a-z]+", re.IGNORECASE
    ),
    re.compile(
        r"\b(?:reuters|associated press|\bap\b|afp|bloomberg news|dow jones)\b",
        re.IGNORECASE,
    ),
]

REPORTING_VERB_PATTERNS = [
    re.compile(
        r"\b(?:said|reported|told|according to|stated|confirmed|announced|explained|added)\b",
        re.IGNORECASE,
    ),
]

NEGATIVE_ECOMMERCE_PATTERNS = [
    re.compile(
        r"\b(?:add to cart|buy now|in stock|product details|price:?\s*[\$\£\€]|free shipping)\b",
        re.IGNORECASE,
    ),
]

NEGATIVE_FORUM_PATTERNS = [
    re.compile(
        r"\b(?:joined:?|posts:?|member since|registered:?|quote reply|thread starter)\b",
        re.IGNORECASE,
    ),
]

ENGLISH_STOPWORDS = {
    "the",
    "be",
    "to",
    "of",
    "and",
    "a",
    "in",
    "that",
    "have",
    "i",
    "it",
    "for",
    "not",
    "on",
    "with",
    "he",
    "as",
    "you",
    "do",
    "at",
    "this",
    "but",
    "his",
    "by",
    "from",
    "they",
    "we",
    "say",
    "her",
    "she",
    "or",
    "an",
    "will",
    "my",
    "one",
    "all",
    "would",
    "there",
    "their",
    "what",
    "so",
    "up",
    "out",
    "if",
    "about",
    "who",
    "get",
    "which",
    "go",
    "me",
    "when",
    "make",
    "can",
    "like",
    "time",
    "no",
    "just",
    "him",
    "know",
    "take",
    "people",
    "into",
    "year",
    "your",
    "good",
    "some",
    "could",
    "them",
    "see",
    "other",
    "than",
    "then",
    "now",
    "look",
    "only",
    "come",
    "its",
    "over",
    "think",
    "also",
}

WORD_REGEX = re.compile(r"\b\w+\b", re.UNICODE)


@dataclass(frozen=True)
class ExtractedARCRecord:
    url: str
    http_status: int
    content_type: str
    html_body: str


class ARCParser:
    """Extracts HTTP headers and HTML payload from compressed ARC byte slices."""

    @staticmethod
    def parse_arc_bytes(compressed_bytes: bytes) -> ExtractedARCRecord | None:
        try:
            decompressed = gzip.decompress(compressed_bytes)
        except Exception:
            return None

        try:
            stream = io.BytesIO(decompressed)
            for record in ArchiveIterator(stream):
                if record.rec_type in {"response", "arc"}:
                    url = (
                        record.rec_headers.get_header("WARC-Target-URI")
                        or record.rec_headers.get_header("ARC-Target-URI")
                        or ""
                    )
                    http_status = int(
                        record.http_headers.get_statuscode()
                        if record.http_headers
                        else 200
                    )
                    content_type = (
                        record.http_headers.get_header("Content-Type")
                        if record.http_headers
                        else "text/html"
                    )
                    payload = record.content_stream().read()
                    html_text = payload.decode("utf-8", errors="replace")
                    return ExtractedARCRecord(
                        url=url,
                        http_status=http_status,
                        content_type=content_type,
                        html_body=html_text,
                    )
        except Exception:
            pass

        # Fallback for plain ARC decompression if ArchiveIterator fails
        return ARCParser._fallback_parse_raw(decompressed)

    @staticmethod
    def _fallback_parse_raw(decompressed: bytes) -> ExtractedARCRecord | None:
        try:
            # ARC header line: URL IP DATE MIME LENGTH
            header_end = decompressed.find(b"\n")
            if header_end == -1:
                return None
            header_line = (
                decompressed[:header_end].decode("utf-8", errors="ignore").strip()
            )
            parts = header_line.split(" ")
            url = parts[0] if parts else ""
            mime = parts[3] if len(parts) > 3 else "text/html"

            body_bytes = decompressed[header_end + 1 :]
            # Check for HTTP header separator \r\n\r\n or \n\n
            sep_idx = body_bytes.find(b"\r\n\r\n")
            if sep_idx != -1:
                html_bytes = body_bytes[sep_idx + 4 :]
            else:
                sep_idx2 = body_bytes.find(b"\n\n")
                html_bytes = (
                    body_bytes[sep_idx2 + 2 :] if sep_idx2 != -1 else body_bytes
                )

            return ExtractedARCRecord(
                url=url,
                http_status=200,
                content_type=mime,
                html_body=html_bytes.decode("utf-8", errors="replace"),
            )
        except Exception:
            return None


class TextExtractor:
    """Extracts main body text from HTML using Trafilatura."""

    @staticmethod
    def extract_text(html: str, url: str | None = None) -> str | None:
        if not html or not html.strip():
            return None
        try:
            text = trafilatura.extract(
                html,
                url=url,
                include_comments=False,
                include_tables=False,
                no_fallback=False,
            )
            return text if text and text.strip() else None
        except Exception:
            return None


class NewsClassifier:
    """Explicit content and metadata news classifier evaluating journalistic cues."""

    @classmethod
    def evaluate(
        cls, text: str, html: str, url: str
    ) -> tuple[float, bool, dict[str, Any]]:
        score = 0.0
        details: dict[str, Any] = {}

        # 1. Structural cues
        has_dateline = any(
            p.search(text) or p.search(html[:1500]) for p in DATELINE_PATTERNS
        )
        has_byline = any(
            p.search(text[:1000]) or p.search(html[:2000]) for p in BYLINE_PATTERNS
        )
        if has_dateline:
            score += 1.5
        if has_byline:
            score += 1.5
        details["has_dateline"] = has_dateline
        details["has_byline"] = has_byline

        # 2. Journalistic reporting discourse
        reporting_verb_count = sum(
            len(p.findall(text)) for p in REPORTING_VERB_PATTERNS
        )
        quote_count = text.count('"') + text.count("“") + text.count("”")
        if reporting_verb_count >= 2:
            score += 1.0
        if quote_count >= 2:
            score += 1.0
        details["reporting_verb_count"] = reporting_verb_count
        details["quote_count"] = quote_count

        # 3. Negative signals
        is_ecommerce = any(p.search(text) for p in NEGATIVE_ECOMMERCE_PATTERNS)
        is_forum = any(p.search(text) for p in NEGATIVE_FORUM_PATTERNS)
        if is_ecommerce:
            score -= 3.0
        if is_forum:
            score -= 3.0
        details["is_ecommerce"] = is_ecommerce
        details["is_forum"] = is_forum

        # Threshold decision (planning baseline: >= 1.5)
        is_news = score >= 1.5
        return score, is_news, details


class LanguageFilter:
    """Verifies English language using langdetect and stopword density."""

    @staticmethod
    def is_english(text: str) -> tuple[bool, float, str]:
        if not text or len(text.split()) < 10:
            return False, 0.0, "unknown"

        # Compute stopword ratio
        words = [w.lower() for w in WORD_REGEX.findall(text)]
        if not words:
            return False, 0.0, "unknown"
        stopword_count = sum(1 for w in words if w in ENGLISH_STOPWORDS)
        stopword_ratio = stopword_count / len(words)

        try:
            lang = langdetect.detect(text[:2000])
            is_en = (lang == "en" and stopword_ratio >= 0.10) or stopword_ratio >= 0.22
            conf = stopword_ratio if is_en else 0.0
            return is_en, conf, lang
        except Exception:
            # Fallback to pure stopword density
            is_en = stopword_ratio >= 0.22
            return is_en, stopword_ratio, "en" if is_en else "unknown"


class ValidityFilter:
    """Applies quality, length, and non-boilerplate filters."""

    @staticmethod
    def check_validity(
        text: str, min_words: int = 100, max_symbol_ratio: float = 0.20
    ) -> tuple[bool, str | None]:
        if not text:
            return False, "empty_text"

        words = WORD_REGEX.findall(text)
        word_count = len(words)
        if word_count < min_words:
            return False, f"too_short_{word_count}_words"

        # Check symbol and punctuation ratio
        chars = len(text)
        non_alphanumeric = sum(1 for c in text if not c.isalnum() and not c.isspace())
        symbol_ratio = non_alphanumeric / chars if chars > 0 else 1.0
        if symbol_ratio > max_symbol_ratio:
            return False, f"high_symbol_ratio_{symbol_ratio:.2f}"

        return True, None


class WordCounter:
    """Feasibility measurement word counting convention."""

    @staticmethod
    def count_words(text: str) -> int:
        return len(WORD_REGEX.findall(text))


@dataclass(frozen=True)
class ProcessedDocumentResult:
    record_id: str
    crawl_id: str
    url: str
    fetch_status: str
    downloaded_bytes: int
    http_status: int
    extraction_success: bool
    clean_text: str | None
    news_score: float
    is_news_predicted: bool
    is_english: bool
    is_valid: bool
    rejection_reason: str | None
    word_count: int
    inclusion_probability: float
    design_weight: float
    proxy_words: int  # y_proxy = is_news * is_en * is_valid * word_count
    diagnostics: dict[str, Any]


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


__all__ = [
    "ARCParser",
    "ExtractedARCRecord",
    "LanguageFilter",
    "NewsClassifier",
    "PipelineRunner",
    "ProcessedDocumentResult",
    "TextExtractor",
    "ValidityFilter",
    "WordCounter",
]
