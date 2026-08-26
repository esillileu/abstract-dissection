"""Tests for TokenBucketLimiter and RangeFetcher."""

from __future__ import annotations

import time

from f2.fetcher import RangeFetcher, TokenBucketLimiter


def test_token_bucket_limiter():
    # Set limit to 10,000 bytes/sec and capacity 5,000
    limiter = TokenBucketLimiter(rate_bytes_per_sec=10000.0, capacity=5000.0)
    limiter.consume(5000)

    # Consume 5,000 bytes (should take approx 0.5 sec)
    start = time.monotonic()
    limiter.consume(5000)
    elapsed = time.monotonic() - start
    assert elapsed >= 0.4  # Verified rate limiting delay


def test_range_fetcher_initialization():
    fetcher = RangeFetcher(bandwidth_mbps=20.0, max_concurrency=2, max_retries=3)
    assert fetcher.base_url == "https://data.commoncrawl.org"
    assert fetcher.max_retries == 3
