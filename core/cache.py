"""Unified cache layer providing bounded caching with metrics.

This module provides a unified cache interface that wraps TTLCache
and other caching backends with metrics tracking and cross-cache invalidation hooks.
"""

import logging
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple
from functools import wraps

from cachetools import TTLCache

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Metrics tracking
# ----------------------------------------------------------------------

_cache_stats: Dict[str, Dict[str, int]] = {}
_cache_stats_lock = threading.Lock()


def _get_cache_stats(cache_name: str) -> Dict[str, int]:
    """Get or create stats for a named cache. Must be called with _cache_stats_lock held."""
    if cache_name not in _cache_stats:
        _cache_stats[cache_name] = {"hits": 0, "misses": 0, "sets": 0, "evictions": 0}
    return _cache_stats[cache_name]


def record_cache_hit(cache_name: str) -> None:
    """Record a cache hit."""
    with _cache_stats_lock:
        _get_cache_stats(cache_name)["hits"] += 1


def record_cache_miss(cache_name: str) -> None:
    """Record a cache miss."""
    with _cache_stats_lock:
        _get_cache_stats(cache_name)["misses"] += 1


def record_cache_set(cache_name: str) -> None:
    """Record a cache set operation."""
    with _cache_stats_lock:
        _get_cache_stats(cache_name)["sets"] += 1


def record_cache_eviction(cache_name: str) -> None:
    """Record a cache eviction."""
    with _cache_stats_lock:
        _get_cache_stats(cache_name)["evictions"] += 1


def get_cache_stats(cache_name: str) -> Dict[str, int]:
    """Get cache statistics for a named cache."""
    with _cache_stats_lock:
        return dict(_get_cache_stats(cache_name))


def get_all_cache_stats() -> Dict[str, Dict[str, int]]:
    """Get statistics for all caches."""
    with _cache_stats_lock:
        return {name: dict(stats) for name, stats in _cache_stats.items()}


# ----------------------------------------------------------------------
# Cache invalidation hooks
# ----------------------------------------------------------------------

_invalidation_hooks: Dict[str, List[Callable[[str], None]]] = {}


def register_invalidation_hook(
    cache_name: str,
    hook: Callable[[str], None],
) -> None:
    """Register a hook to be called when a cache key is invalidated."""
    if cache_name not in _invalidation_hooks:
        _invalidation_hooks[cache_name] = []
    _invalidation_hooks[cache_name].append(hook)


def notify_invalidation(cache_name: str, key: str) -> None:
    """Notify all hooks that a key has been invalidated."""
    if cache_name in _invalidation_hooks:
        for hook in _invalidation_hooks[cache_name]:
            try:
                hook(key)
            except Exception:
                logger.debug(f"Invalidation hook error for {cache_name}:{key}")


# ----------------------------------------------------------------------
# Unified cache creation
# ----------------------------------------------------------------------


def create_ttl_cache(
    name: str,
    maxsize: int = 1000,
    ttl: int = 3600,
) -> TTLCache:
    """
    Create a TTLCache with metrics tracking wrapper.
    
    Args:
        name: Cache name for metrics tracking
        maxsize: Maximum number of entries
        ttl: Time-to-live in seconds
        
    Returns:
        TTLCache wrapped with metrics tracking
    """
    cache = TTLCache(maxsize=maxsize, ttl=ttl)
    return MetricsTTLCache(name, cache)


class MetricsTTLCache:
    """TTLCache wrapper that tracks hit/miss/set/eviction metrics."""

    def __init__(self, name: str, wrapped: TTLCache):
        self.name = name
        self._wrapped = wrapped

    def __contains__(self, key: Any) -> bool:
        result = key in self._wrapped
        if result:
            record_cache_hit(self.name)
        else:
            record_cache_miss(self.name)
        return result

    def __getitem__(self, key: Any) -> Any:
        try:
            value = self._wrapped[key]
            record_cache_hit(self.name)
            return value
        except KeyError:
            record_cache_miss(self.name)
            raise

    def __setitem__(self, key: Any, value: Any) -> None:
        old_len = len(self._wrapped)
        self._wrapped[key] = value
        record_cache_set(self.name)
        # Check if eviction occurred
        if len(self._wrapped) < old_len:
            record_cache_eviction(self.name)

    def get(self, key: Any, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def pop(self, key: Any, *args: Any) -> Any:
        try:
            value = self._wrapped.pop(key)
            notify_invalidation(self.name, str(key))
            return value
        except KeyError:
            if args:
                return args[0]
            raise

    def __delitem__(self, key: Any) -> None:
        del self._wrapped[key]
        notify_invalidation(self.name, str(key))

    def clear(self) -> None:
        """Clear the cache and notify all hooks."""
        for key in list(self._wrapped.keys()):
            notify_invalidation(self.name, str(key))
        self._wrapped.clear()

    def __len__(self) -> int:
        return len(self._wrapped)

    def keys(self):
        return self._wrapped.keys()

    def values(self):
        return self._wrapped.values()

    def items(self):
        return self._wrapped.items()

    @property
    def maxsize(self) -> int:
        return self._wrapped.maxsize

    @property
    def ttl(self) -> float:
        return self._wrapped.ttl


# ----------------------------------------------------------------------
# Decorator for cached functions
# ----------------------------------------------------------------------


def cached(
    name: str,
    ttl: int = 3600,
    maxsize: int = 1000,
    key_func: Optional[Callable[..., Tuple]] = None,
):
    """
    Decorator that caches async function results with metrics.
    
    Usage:
        @cached("my_cache", ttl=60, maxsize=100)
        async def my_func(arg1, arg2):
            ...
    """
    def decorator(func: Callable):
        cache = create_ttl_cache(name, maxsize=maxsize, ttl=ttl)

        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Build cache key
            if key_func:
                key = key_func(*args, **kwargs)
            else:
                key = (func.__name__, args, tuple(sorted(kwargs.items())))

            # Check cache
            if key in cache:
                return cache[key]

            # Call function
            result = await func(*args, **kwargs)

            # Store in cache
            cache[key] = result
            return result

        return wrapper
    return decorator


# ----------------------------------------------------------------------
# Convenience: adapt an existing TTLCache
# ----------------------------------------------------------------------


def adapt_existing_ttl(name: str, existing_cache: TTLCache) -> MetricsTTLCache:
    """Adapt an existing TTLCache with metrics tracking."""
    return MetricsTTLCache(name, existing_cache)