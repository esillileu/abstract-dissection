"""Pure processing pipeline: ARC parsing, Trafilatura extraction, news classification, filters, and word counting."""

from __future__ import annotations

from .classifier import (
    BYLINE_PATTERNS,
    DATELINE_PATTERNS,
    NEGATIVE_ECOMMERCE_PATTERNS,
    NEGATIVE_FORUM_PATTERNS,
    REPORTING_VERB_PATTERNS,
    NewsClassifier,
)
from .extractor import TextExtractor
from .filters import (
    ENGLISH_STOPWORDS,
    WORD_REGEX,
    LanguageFilter,
    ValidityFilter,
    WordCounter,
)
from .models import ProcessedDocumentResult
from .runner import PipelineRunner

__all__ = [
    "BYLINE_PATTERNS",
    "DATELINE_PATTERNS",
    "ENGLISH_STOPWORDS",
    "NEGATIVE_ECOMMERCE_PATTERNS",
    "NEGATIVE_FORUM_PATTERNS",
    "REPORTING_VERB_PATTERNS",
    "WORD_REGEX",
    "LanguageFilter",
    "NewsClassifier",
    "PipelineRunner",
    "ProcessedDocumentResult",
    "TextExtractor",
    "ValidityFilter",
    "WordCounter",
]
