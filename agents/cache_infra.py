"""Cache infrastructure for planner - TTLCache-based caching with per-key locking."""

import asyncio
import time
from functools import wraps
from typing import Any, Callable, Dict

from cachetools import TTLCache

DISABLE_CACHE = False  # can be toggled for dev/testing

_cache_locks: Dict[Any, asyncio.Lock] = {}
_locks_lock = asyncio.Lock()


class CacheLock(asyncio.Lock):
    """Lock for cache entries - used as a marker class."""
    pass


async def _get_cache_lock(key: Any) -> asyncio.Lock:
    """Get or create a lock for a specific cache key to prevent cache stampede."""
    global _cache_locks
    if key not in _cache_locks:
        async with _locks_lock:
            if key not in _cache_locks:
                _cache_locks[key] = CacheLock()
    return _cache_locks[key]


def async_cache(ttl: int, maxsize: int = 1000):
    """
    Decorator that caches the result of an async function for `ttl` seconds.
    Uses per-key locks to prevent cache stampede and bounded cache to limit memory.
    """
    def decorator(func):
        # Use TTLCache if available, else simple dict (unbounded)
        if TTLCache:
            cache = TTLCache(maxsize=maxsize, ttl=ttl)
        else:
            cache = {}
            import logging
            logging.getLogger(__name__).warning(
                "cachetools not installed, using unbounded cache (memory may grow)"
            )

        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Build cache key
            key = (func.__name__, args, frozenset(kwargs.items()))
            lock = await _get_cache_lock(key)

            async with lock:
                if not DISABLE_CACHE:  # skip entire cache block in dev
                    if TTLCache:
                        if key in cache:
                            return cache[key]
                    else:
                        now = time.monotonic()
                        if key in cache:
                            result, timestamp = cache[key]
                            if now - timestamp < ttl:
                                return result

            result = await func(*args, **kwargs)

            if not DISABLE_CACHE:  # don't store if cache disabled
                if TTLCache:
                    cache[key] = result
                else:
                    cache[key] = (result, time.monotonic())

            return result

        return wrapper
    return decorator


def create_cached_fetcher(ttl: int, maxsize: int, fetch_func: Callable):
    """
    Create an async cached version of a fetch function.
    The fetch function must accept the same arguments each time.
    """
    @async_cache(ttl=ttl, maxsize=maxsize)
    async def cached(*args, **kwargs):
        return await fetch_func(*args, **kwargs)
    return cached