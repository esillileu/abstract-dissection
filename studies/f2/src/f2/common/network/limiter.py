"""Thread-safe Token Bucket rate limiter to restrict network bandwidth consumption."""

from __future__ import annotations

import threading
import time


class TokenBucketLimiter:
    """Thread-safe Token Bucket rate limiter to restrict network bandwidth consumption."""

    def __init__(
        self, rate_bytes_per_sec: float, capacity: float | None = None
    ) -> None:
        self.rate = rate_bytes_per_sec
        self.capacity = (
            capacity if capacity is not None else max(rate_bytes_per_sec, 65536.0)
        )
        self.tokens = self.capacity
        self.last_update = time.monotonic()
        self.lock = threading.Lock()

    @classmethod
    def from_mbps(cls, mbps: float) -> TokenBucketLimiter:
        bytes_per_sec = (mbps * 1_000_000.0) / 8.0
        return cls(bytes_per_sec)

    def consume(self, num_bytes: int) -> None:
        if self.rate <= 0:
            return

        while True:
            with self.lock:
                now = time.monotonic()
                elapsed = now - self.last_update
                self.last_update = now
                self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)

                if self.tokens >= num_bytes:
                    self.tokens -= num_bytes
                    return
                # Need to wait
                needed = num_bytes - self.tokens
                wait_time = needed / self.rate

            time.sleep(min(wait_time, 0.5))


__all__ = ["TokenBucketLimiter"]
