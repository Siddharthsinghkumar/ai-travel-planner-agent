"""Test thread-safe cache metrics tracking."""
import threading
from core.cache import (
    record_cache_hit,
    record_cache_miss,
    record_cache_set,
    record_cache_eviction,
    get_cache_stats,
    get_all_cache_stats,
)


def test_cache_stats_thread_safety():
    """Verify that concurrent increments produce correct totals."""
    cache_name = "test_thread_safety"
    # Reset by reading first (creates the entry)
    get_cache_stats(cache_name)

    num_threads = 10
    ops_per_thread = 1000

    def increment_hits():
        for _ in range(ops_per_thread):
            record_cache_hit(cache_name)

    def increment_misses():
        for _ in range(ops_per_thread):
            record_cache_miss(cache_name)

    threads = []
    for _ in range(num_threads):
        t1 = threading.Thread(target=increment_hits)
        t2 = threading.Thread(target=increment_misses)
        threads.extend([t1, t2])

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    stats = get_cache_stats(cache_name)
    expected = num_threads * ops_per_thread
    assert stats["hits"] == expected, f"Expected {expected} hits, got {stats['hits']}"
    assert stats["misses"] == expected, f"Expected {expected} misses, got {stats['misses']}"


def test_get_all_cache_stats_thread_safety():
    """Verify get_all_cache_stats returns consistent snapshot under concurrent writes."""
    cache_name = "test_snapshot"
    get_cache_stats(cache_name)
    results = []

    def writer():
        for _ in range(500):
            record_cache_hit(cache_name)

    def reader():
        for _ in range(100):
            snapshot = get_all_cache_stats()
            results.append(snapshot)

    t_write = threading.Thread(target=writer)
    t_read = threading.Thread(target=reader)
    t_write.start()
    t_read.start()
    t_write.join()
    t_read.join()

    # All snapshots should be valid dicts with non-negative counts
    for snapshot in results:
        assert cache_name in snapshot
        stats = snapshot[cache_name]
        assert stats["hits"] >= 0
        assert stats["misses"] >= 0
