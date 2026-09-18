"""Seed domain catalog, URL patterns, and news path heuristics."""

from __future__ import annotations

import re
from enum import StrEnum


class DomainStratum(StrEnum):
    GLOBAL = "global_agencies"
    NATIONAL = "national_press"
    REGIONAL = "regional_press"
    SPECIALTY = "specialty_media"
    SYNDICATED = "syndicated_outlets"


SEED_DOMAIN_CATALOG: dict[DomainStratum, list[str]] = {
    DomainStratum.GLOBAL: [
        "reuters.com",
        "bbc.co.uk",
        "ap.org",
        "bloomberg.com",
        "afp.com",
    ],
    DomainStratum.NATIONAL: [
        "nytimes.com",
        "washingtonpost.com",
        "theguardian.com",
        "telegraph.co.uk",
        "smh.com.au",
        "wsj.com",
        "usatoday.com",
        "independent.co.uk",
    ],
    DomainStratum.REGIONAL: [
        "chicagotribune.com",
        "sfgate.com",
        "seattletimes.com",
        "boston.com",
        "latimes.com",
        "startribune.com",
        "denverpost.com",
    ],
    DomainStratum.SPECIALTY: [
        "techcrunch.com",
        "wired.com",
        "arstechnica.com",
        "economist.com",
        "forbes.com",
        "cnet.com",
        "venturebeat.com",
    ],
    DomainStratum.SYNDICATED: [
        "prnewswire.com",
        "businesswire.com",
        "marketwatch.com",
        "upi.com",
    ],
}

NEWS_PATH_PATTERNS = [
    re.compile(r"/(?:news|article|articles|story|stories)/", re.IGNORECASE),
    re.compile(
        r"/(?:world|politics|business|technology|science|opinion)/\d{4}/", re.IGNORECASE
    ),
    re.compile(r"/\d{4}/\d{1,2}/\d{1,2}/[a-z0-9\-]+", re.IGNORECASE),
    re.compile(r"/[a-z0-9\-]+-\d{5,}\.html?$", re.IGNORECASE),
]

# Exact 32-extension Rule 1 regex from 10k calibration baseline
ALL_BINARY_EXT = re.compile(
    r"\.(jpg|jpeg|png|gif|css|js|pdf|mp3|mp4|avi|zip|gz|tar|tgz|exe|dmg|iso|bin|doc|docx|ppt|pptx|xls|xlsx|rss|xml|json|swf|ico|woff|ttf|svg)(\?.*)?$",
    re.IGNORECASE,
)


def is_news_path_heuristic(url: str) -> bool:
    """Evaluate whether URL matches standard news path heuristics."""
    return any(pattern.search(url) is not None for pattern in NEWS_PATH_PATTERNS)
