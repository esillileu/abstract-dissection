"""Pipeline data models and result contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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


__all__ = ["ProcessedDocumentResult"]
