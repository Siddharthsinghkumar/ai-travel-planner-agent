"""In-process sliding-window rate limiting for single-node runtime."""

from __future__ import annotations

import asyncio
import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict


@dataclass(frozen=True)
class RateLimitDecision:
    allowed: bool
    retry_after_seconds: int
    remaining: int
    limit: int
    window_seconds: int


class SlidingWindowRateLimiter:
    def __init__(self, *, max_keys: int = 10_000) -> None:
        self._events: Dict[str, Deque[float]] = {}
        self._lock = asyncio.Lock()
        self._max_keys = max(100, int(max_keys))

    async def check(self, key: str, *, limit: int, window_seconds: int) -> RateLimitDecision:
        normalized_key = str(key or "").strip() or "anonymous"
        normalized_limit = max(1, int(limit))
        normalized_window = max(1, int(window_seconds))

        async with self._lock:
            now = time.monotonic()
            bucket = self._events.setdefault(normalized_key, deque())
            cutoff = now - normalized_window
            while bucket and bucket[0] <= cutoff:
                bucket.popleft()

            if len(bucket) >= normalized_limit:
                retry_after = max(1, int(math.ceil((bucket[0] + normalized_window) - now)))
                return RateLimitDecision(
                    allowed=False,
                    retry_after_seconds=retry_after,
                    remaining=0,
                    limit=normalized_limit,
                    window_seconds=normalized_window,
                )

            bucket.append(now)
            remaining = max(0, normalized_limit - len(bucket))
            self._trim_keys_locked()
            return RateLimitDecision(
                allowed=True,
                retry_after_seconds=0,
                remaining=remaining,
                limit=normalized_limit,
                window_seconds=normalized_window,
            )

    def _trim_keys_locked(self) -> None:
        if len(self._events) <= self._max_keys:
            return
        empty_keys = [key for key, values in self._events.items() if not values]
        for key in empty_keys:
            self._events.pop(key, None)
        if len(self._events) <= self._max_keys:
            return

        # Drop oldest buckets first when over capacity.
        overflow = len(self._events) - self._max_keys
        oldest = sorted(self._events.items(), key=lambda item: item[1][-1] if item[1] else -1.0)[:overflow]
        for key, _ in oldest:
            self._events.pop(key, None)
