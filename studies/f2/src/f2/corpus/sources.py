"""Frozen source and validation specifications for the canonical F2 corpus."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class SourceFile:
    name: str
    url: str
    year: int | None = None
    sha256: str | None = None


@dataclass(frozen=True)
class CorpusSource:
    key: str
    release: str
    name: str
    homepage: str
    license: str
    access: str
    files: tuple[SourceFile, ...]
    raw_resource_version_id: str
    canonical_resource_version_id: str
    normalized_resource_version_id: str
    blocked_reason: str | None = None

    @property
    def config_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


_WMT_ROOT = "https://www.statmt.org/wmt14/training-monolingual-news-crawl"

SOURCES: tuple[CorpusSource, ...] = (
    CorpusSource(
        key="lm1b",
        release="r13output",
        name="One Billion Word Language Modeling Benchmark",
        homepage="https://www.statmt.org/lm-benchmark/",
        license="release terms at source",
        access="public",
        files=(
            SourceFile(
                "1-billion-word-language-modeling-benchmark-r13output.tar.gz",
                "https://www.statmt.org/lm-benchmark/1-billion-word-language-modeling-benchmark-r13output.tar.gz",
                sha256="01ba60381110baf7f189dfd2b8374de371e8c9a340835793f190bdae9e90a34e",
            ),
        ),
        raw_resource_version_id="f2-lm1b-r13output-raw",
        canonical_resource_version_id="f2-lm1b-r13output-canonical-v1",
        normalized_resource_version_id="f2-lm1b-r13output-normalized-v1",
    ),
    CorpusSource(
        key="wmt",
        release="news-crawl-2007-2012",
        name="WMT News Crawl English 2007-2012",
        homepage=f"{_WMT_ROOT}/",
        license="WMT release terms at source",
        access="public",
        files=tuple(
            SourceFile(
                f"news.{year}.en.shuffled.gz",
                f"{_WMT_ROOT}/news.{year}.en.shuffled.gz",
                year,
            )
            for year in range(2007, 2013)
        ),
        raw_resource_version_id="f2-wmt-news-2007-2012-raw",
        canonical_resource_version_id="f2-wmt-news-2007-2012-canonical-v1",
        normalized_resource_version_id="f2-wmt-news-2007-2012-normalized-v1",
    ),
    CorpusSource(
        key="gigaword",
        release="LDC2011T07",
        name="English Gigaword Fifth Edition",
        homepage="https://catalog.ldc.upenn.edu/LDC2011T07",
        license="LDC license required",
        access="licensed",
        files=(),
        raw_resource_version_id="f2-gigaword5-ldc2011t07-raw",
        canonical_resource_version_id="f2-gigaword5-canonical-v1",
        normalized_resource_version_id="f2-gigaword5-normalized-v1",
        blocked_reason="authorized LDC access required",
    ),
    CorpusSource(
        key="umbc",
        release="2013",
        name="UMBC WebBase Corpus",
        homepage="https://ebiquity.umbc.edu/resource/html/id/351/UMBC-webbase-corpus",
        license="CC BY 3.0",
        access="public",
        files=(
            SourceFile(
                "umbc_webbase_corpus.tgz",
                "https://ebiquity.umbc.edu/share/umbc_webbase_corpus.tgz",
            ),
        ),
        raw_resource_version_id="f2-umbc-webbase-2013-raw",
        canonical_resource_version_id="f2-umbc-webbase-canonical-v1",
        normalized_resource_version_id="f2-umbc-webbase-normalized-v1",
    ),
    CorpusSource(
        key="wikipedia",
        release="2012-12-01",
        name="English Wikipedia pages-articles dump",
        homepage="https://dumps.wikimedia.org/enwiki/20121201/",
        license="CC BY-SA / GFDL",
        access="public-checksum-required",
        files=(
            SourceFile(
                "enwiki-20121201-pages-articles.xml.bz2",
                "https://dumps.wikimedia.org/enwiki/20121201/enwiki-20121201-pages-articles.xml.bz2",
            ),
        ),
        raw_resource_version_id="f2-enwiki-20121201-raw",
        canonical_resource_version_id="f2-enwiki-20121201-canonical-v1",
        normalized_resource_version_id="f2-enwiki-20121201-normalized-v1",
    ),
)

SOURCE_BY_KEY = {source.key: source for source in SOURCES}

VALIDATION_PROFILES = {
    "raw-integrity-v1": {
        "revision": 1,
        "checks": ["expected_length", "sha256_after_upload"],
    },
    "transformation-v1": {
        "revision": 1,
        "checks": [
            "recipe_hash",
            "logical_sha256",
            "record_boundaries",
            "ordered_manifest",
        ],
    },
    "word2vec-2013-compatibility-v1": {
        "revision": 1,
        "checks": ["normalize_text_parity", "domain_mismatch", "time_mismatch"],
        "maximum_verdict": "compatible_reconstruction",
    },
}


SOURCE_BOUNDARY_POLICIES: dict[str, str] = {
    "lm1b": "sentence_per_line",
    "wmt": "sentence_per_line",
    "umbc": "document_paragraph_lines",
    "wikipedia": "article_paragraph_lines",
    "gigaword": "article_paragraph_lines",
}


def stable_id(*parts: object, length: int = 32) -> str:
    value = "\0".join(str(part) for part in parts)
    return hashlib.sha256(value.encode()).hexdigest()[:length]


__all__ = [
    "SOURCES",
    "SOURCE_BOUNDARY_POLICIES",
    "SOURCE_BY_KEY",
    "VALIDATION_PROFILES",
    "CorpusSource",
    "SourceFile",
    "stable_id",
]
