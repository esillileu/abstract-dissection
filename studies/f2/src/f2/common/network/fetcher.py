"""Bounded, rate-limited HTTP Range GET fetcher and result dataclass."""

from __future__ import annotations

import random
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from .limiter import TokenBucketLimiter


@dataclass(frozen=True)
class FetchResult:
    status_code: int
    data: bytes
    downloaded_bytes: int
    elapsed_sec: float
    error_message: str | None = None


class RangeFetcher:
    """Bounded, rate-limited HTTP Range GET fetcher."""

    def __init__(
        self,
        base_url: str = "https://data.commoncrawl.org",
        bandwidth_mbps: float = 20.0,
        max_concurrency: int = 2,
        max_retries: int = 5,
        timeout_sec: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.limiter = TokenBucketLimiter.from_mbps(bandwidth_mbps)
        self.semaphore = threading.Semaphore(max_concurrency)
        self.max_retries = max_retries
        self.timeout = timeout_sec

    def fetch_range(self, url_or_path: str, offset: int, length: int) -> FetchResult:
        """Fetch exact byte range [offset, offset + length - 1]."""
        if url_or_path.startswith("http://") or url_or_path.startswith("https://"):
            url = url_or_path
        else:
            url = f"{self.base_url}/{url_or_path.lstrip('/')}"

        range_header = f"bytes={offset}-{offset + length - 1}"
        headers = {
            "Range": range_header,
            "User-Agent": "abstract-dissection-repro/0.1 (Research reproduction study)",
            "Accept-Encoding": "identity",
        }

        start_time = time.monotonic()
        last_err = None

        for attempt in range(self.max_retries):
            with self.semaphore:
                try:
                    req = urllib.request.Request(url, headers=headers)
                    with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                        data = resp.read()
                        downloaded = len(data)
                        self.limiter.consume(downloaded)
                        elapsed = time.monotonic() - start_time
                        return FetchResult(
                            status_code=resp.status,
                            data=data,
                            downloaded_bytes=downloaded,
                            elapsed_sec=elapsed,
                        )
                except urllib.error.HTTPError as exc:
                    last_err = f"HTTP {exc.code}: {exc.reason}"
                    if exc.code in {429, 500, 502, 503, 504}:
                        backoff = (2**attempt) * 0.5 + random.uniform(0.1, 0.5)
                        time.sleep(backoff)
                        continue
                    # Non-retryable HTTP error (e.g. 404)
                    break
                except Exception as exc:
                    last_err = str(exc)
                    backoff = (2**attempt) * 0.5 + random.uniform(0.1, 0.5)
                    time.sleep(backoff)

        elapsed = time.monotonic() - start_time
        return FetchResult(
            status_code=500,
            data=b"",
            downloaded_bytes=0,
            elapsed_sec=elapsed,
            error_message=last_err or "Unknown fetch error",
        )


__all__ = [
    "FetchResult",
    "RangeFetcher",
]
