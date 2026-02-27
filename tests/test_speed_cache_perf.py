
import time
import pytest
from src.core.speed_cache import SpeedCache

def test_speed_cache_stream_correctness():
    """Verify that the optimized get_stream returns the correct items."""
    cache = SpeedCache()
    cache.clear()

    stream_name = "test_stream"
    items = list(range(100))

    # Push items
    for item in items:
        cache.push_stream(stream_name, item, max_len=100)

    # Test full retrieval
    retrieved = cache.get_stream(stream_name, limit=100)
    assert len(retrieved) == 100
    assert [x["data"] for x in retrieved] == items

    # Test partial retrieval (last 10)
    retrieved_partial = cache.get_stream(stream_name, limit=10)
    assert len(retrieved_partial) == 10
    assert [x["data"] for x in retrieved_partial] == items[-10:]

    # Test limit > size
    retrieved_large = cache.get_stream(stream_name, limit=200)
    assert len(retrieved_large) == 100
    assert [x["data"] for x in retrieved_large] == items

def test_speed_cache_stream_performance():
    """Benchmark the get_stream performance."""
    cache = SpeedCache()
    cache.clear()

    stream_name = "perf_stream"
    N = 50000
    # Push many items
    for i in range(N):
        cache.push_stream(stream_name, i, max_len=N)

    start_time = time.perf_counter()
    # Retrieve small slice from large stream
    _ = cache.get_stream(stream_name, limit=10)
    end_time = time.perf_counter()

    duration = end_time - start_time
    print(f"\nTime to retrieve 10 items from {N}: {duration:.6f}s")

    # We expect this to be very fast (< 1ms) because of O(k) optimization
    # Without optimization, it would be slower due to O(N) copy
    assert duration < 0.01, "get_stream took too long!"

if __name__ == "__main__":
    test_speed_cache_stream_correctness()
    test_speed_cache_stream_performance()
