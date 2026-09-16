from __future__ import annotations

import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

from repro_io.checksum import sha256_file

CHUNK_SIZE = 1024 * 1024


class BandwidthScheduler:
    """Time-aware token-bucket bandwidth limiter.

    Enforces separate limits for peak and off-peak hours using a simple
    token-bucket algorithm.  Tokens refill continuously; ``drain`` blocks
    until the requested number of bytes can be sent.

    Args:
        peak_mbps:        Limit (Mbit/s) during peak hours (09:00-22:00 local).
        offpeak_mbps:     Limit (Mbit/s) during off-peak hours (22:00-09:00 local).
        peak_start:       Hour (0-23) at which the peak window starts (default 9).
        peak_end:         Hour (0-23) at which the peak window ends (default 22).
        current_hour_fn:  Optional function returning the current hour (0-23).
        time_fn:          Optional monotonic clock function returning float seconds.
        sleep_fn:         Optional sleep function taking float seconds.
    """

    def __init__(
        self,
        peak_mbps: float,
        offpeak_mbps: float,
        *,
        peak_start: int = 9,
        peak_end: int = 22,
        current_hour_fn: Callable[[], int] | None = None,
        time_fn: Callable[[], float] | None = None,
        sleep_fn: Callable[[float], None] | None = None,
    ) -> None:
        if peak_mbps <= 0 or offpeak_mbps <= 0:
            raise ValueError("bandwidth limits must be positive")
        self.peak_mbps = peak_mbps
        self.offpeak_mbps = offpeak_mbps
        self.peak_start = peak_start
        self.peak_end = peak_end
        self._current_hour_fn = (
            current_hour_fn
            if current_hour_fn is not None
            else lambda: time.localtime().tm_hour
        )
        self._time_fn = time_fn if time_fn is not None else time.monotonic
        self._sleep_fn = sleep_fn if sleep_fn is not None else time.sleep
        # Token bucket state
        self._tokens: float = 0.0
        self._last_refill: float = self._time_fn()

    def limit_bps(self) -> float:
        hour = self._current_hour_fn()
        in_peak = self.peak_start <= hour < self.peak_end
        mbps = self.peak_mbps if in_peak else self.offpeak_mbps
        return mbps * 125_000.0  # Mbit/s → bytes/s

    def drain(self, nbytes: int) -> None:
        """Block until *nbytes* tokens are available, then consume them."""
        now = self._time_fn()
        elapsed = now - self._last_refill
        limit = self.limit_bps()
        self._tokens = min(self._tokens + elapsed * limit, limit)
        self._last_refill = now

        if self._tokens >= nbytes:
            self._tokens -= nbytes
            return

        # Need to wait for more tokens
        deficit = nbytes - self._tokens
        wait = deficit / limit
        self._sleep_fn(wait)
        self._tokens = 0.0
        self._last_refill = self._time_fn()


class SerialDownloader:
    """HTTP downloader with Range resume, bounded retry, integrity checks,
    and optional time-aware bandwidth throttling.

    Args:
        retries:           Maximum retry attempts after the first failure.
        backoff_seconds:   Base sleep time for exponential backoff.
        timeout:           Socket timeout in seconds per read.
        bandwidth:         Optional :class:`BandwidthScheduler` instance.
                           Pass ``None`` (default) for unlimited throughput.
    """

    def __init__(
        self,
        retries: int = 4,
        backoff_seconds: float = 1.0,
        timeout: float = 60.0,
        bandwidth: BandwidthScheduler | None = None,
        user_agent: str = "repro-io/0.1",
    ) -> None:
        self.user_agent = user_agent
        self.retries = retries
        self.backoff_seconds = backoff_seconds
        self.timeout = timeout
        self.bandwidth = bandwidth

    def download(
        self,
        url: str,
        destination: Path,
        *,
        expected_sha256: str | None = None,
        expected_length: int | None = None,
    ) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_name(destination.name + ".part")
        for attempt in range(self.retries + 1):
            offset = partial.stat().st_size if partial.exists() else 0
            headers = {"User-Agent": self.user_agent}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            try:
                with urllib.request.urlopen(
                    urllib.request.Request(url, headers=headers), timeout=self.timeout
                ) as response:
                    status = getattr(response, "status", 200)
                    if offset and status != 206:
                        partial.unlink()
                        offset = 0
                    mode = "ab" if offset else "wb"
                    with partial.open(mode) as output:
                        while chunk := response.read(CHUNK_SIZE):
                            if self.bandwidth is not None:
                                self.bandwidth.drain(len(chunk))
                            output.write(chunk)
                size = partial.stat().st_size
                if expected_length is not None and size != expected_length:
                    raise ValueError(
                        f"length mismatch: expected {expected_length}, observed {size}"
                    )
                observed = sha256_file(partial)
                if expected_sha256 is not None and observed != expected_sha256.lower():
                    raise ValueError(
                        f"SHA-256 mismatch: expected {expected_sha256}, observed {observed}"
                    )
                partial.replace(destination)
                return destination
            except (OSError, urllib.error.URLError, ValueError):
                if attempt == self.retries:
                    raise
                time.sleep(self.backoff_seconds * (2**attempt))
        raise AssertionError("unreachable")
