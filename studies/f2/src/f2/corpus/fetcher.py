"""Bounded, rate-limited HTTP Range GET fetcher for Common Crawl archives."""

from __future__ import annotations

from f2.common.network.fetcher import FetchResult
from f2.common.network.fetcher import RangeFetcher as BaseRangeFetcher
from f2.common.network.limiter import TokenBucketLimiter

from .cdx import CDXBlockLocator


class RangeFetcher(BaseRangeFetcher):
    """Common Crawl specialized RangeFetcher."""

    def fetch_cdx_block(self, crawl_id: str, block: CDXBlockLocator) -> FetchResult:
        """Fetch a specific CDX block slice from data.commoncrawl.org."""
        filename = f"cc-index/collections/{crawl_id}/indexes/{block.filename}"
        return self.fetch_range(filename, block.offset, block.length)


__all__ = [
    "FetchResult",
    "RangeFetcher",
    "TokenBucketLimiter",
]
