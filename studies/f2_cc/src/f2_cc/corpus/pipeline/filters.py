"""Language verification, validity filtering, and word counting."""

from __future__ import annotations

import re

import langdetect

ENGLISH_STOPWORDS = {
    "a",
    "about",
    "all",
    "also",
    "an",
    "and",
    "as",
    "at",
    "be",
    "but",
    "by",
    "can",
    "come",
    "could",
    "do",
    "for",
    "from",
    "get",
    "go",
    "good",
    "have",
    "he",
    "her",
    "him",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "just",
    "know",
    "like",
    "look",
    "make",
    "me",
    "my",
    "no",
    "not",
    "now",
    "of",
    "on",
    "one",
    "only",
    "or",
    "other",
    "our",
    "out",
    "over",
    "people",
    "say",
    "see",
    "she",
    "so",
    "some",
    "take",
    "than",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "think",
    "this",
    "time",
    "to",
    "two",
    "up",
    "us",
    "use",
    "was",
    "we",
    "what",
    "when",
    "which",
    "who",
    "will",
    "with",
    "would",
    "year",
    "you",
    "your",
}

WORD_REGEX = re.compile(r"\b\w+\b", re.UNICODE)


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


__all__ = [
    "ENGLISH_STOPWORDS",
    "WORD_REGEX",
    "LanguageFilter",
    "ValidityFilter",
    "WordCounter",
]
