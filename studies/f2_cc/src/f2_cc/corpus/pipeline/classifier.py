"""Explicit content and metadata news classifier evaluating journalistic cues."""

from __future__ import annotations

import re
from typing import Any

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


__all__ = [
    "BYLINE_PATTERNS",
    "DATELINE_PATTERNS",
    "NEGATIVE_ECOMMERCE_PATTERNS",
    "NEGATIVE_FORUM_PATTERNS",
    "REPORTING_VERB_PATTERNS",
    "NewsClassifier",
]
