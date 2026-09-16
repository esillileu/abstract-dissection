"""Bounded, rate-limited HTTP Range GET fetcher for Common Crawl archives."""

from __future__ import annotations

from repro_io.http.fetcher import FetchResult
from repro_io.http.fetcher import RangeFetcher as BaseRangeFetcher
from repro_io.http.limiter import TokenBucketLimiter

from .cdx import CDXBlockLocator


class RangeFetcher(BaseRangeFetcher):
    """Common Crawl specialized RangeFetcher."""

    def __init__(self, base_url: str = "https://data.commoncrawl.org", **kwargs):
        super().__init__(base_url=base_url, **kwargs)

    def fetch_cdx_block(self, crawl_id: str, block: CDXBlockLocator) -> FetchResult:
        """Fetch a specific CDX block slice from data.commoncrawl.org."""
        filename = f"cc-index/collections/{crawl_id}/indexes/{block.filename}"
        return self.fetch_range(filename, block.offset, block.length)


__all__ = [
    "FetchResult",
    "RangeFetcher",
    "TokenBucketLimiter",
]
