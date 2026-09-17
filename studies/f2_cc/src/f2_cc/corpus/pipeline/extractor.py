"""HTML body text extraction using Trafilatura."""

from __future__ import annotations

import trafilatura


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


__all__ = ["TextExtractor"]
