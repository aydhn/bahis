import time
import random
from timeit import default_timer as timer

class OldTTLCache:
    def __init__(self, ttl_seconds=3600.0, max_size=500):
        self._ttl = ttl_seconds
        self._max_size = max_size
        self._store = {}

    def set(self, key, value):
        if len(self._store) >= self._max_size:
            self._evict_oldest()
        self._store[key] = (time.time(), value)

    def _evict_oldest(self):
        now = time.time()
        expired = [k for k, (ts, _) in self._store.items() if now - ts > self._ttl]
        for k in expired:
            del self._store[k]

        if len(self._store) >= self._max_size:
            oldest_key = min(self._store, key=lambda k: self._store[k][0])
            del self._store[oldest_key]

class NewTTLCache:
    def __init__(self, ttl_seconds=3600.0, max_size=500):
        self._ttl = ttl_seconds
        self._max_size = max_size
        self._store = {}

    def set(self, key, value):
        # Update existing key or insert new.
        # Removing and re-inserting ensures the key is moved to the end of the dictionary
        # (in Python 3.7+ dictionaries, preserving insertion order)
        if key in self._store:
            del self._store[key]
        elif len(self._store) >= self._max_size:
            self._evict_oldest()

        self._store[key] = (time.time(), value)

    def _evict_oldest(self):
        now = time.time()

        # Since Python 3.7+, dicts maintain insertion order.
        # The oldest items are at the beginning of the dictionary.
        # We can iterate from the beginning and delete expired ones.
        keys_to_del = []
        for k, (ts, _) in self._store.items():
            if now - ts > self._ttl:
                keys_to_del.append(k)
            else:
                # Assuming entries are inserted in chronological order,
                # if one isn't expired, subsequent ones aren't either.
                break

        for k in keys_to_del:
            del self._store[k]

        # If we are STILL at max capacity after purging expired items,
        # we drop the oldest remaining item (the first one in the dict).
        if len(self._store) >= self._max_size:
            oldest_key = next(iter(self._store))
            del self._store[oldest_key]


def benchmark():
    keys = [f"key_{i}" for i in range(50000)]

    start = timer()
    old_cache = OldTTLCache(max_size=1000)
    for k in keys:
        old_cache.set(k, 1)
    old_time = timer() - start

    start = timer()
    new_cache = NewTTLCache(max_size=1000)
    for k in keys:
        new_cache.set(k, 1)
    new_time = timer() - start

    print(f"Old time: {old_time:.4f}s")
    print(f"New time: {new_time:.4f}s")
    print(f"Speedup: {old_time / new_time:.2f}x")

if __name__ == "__main__":
    benchmark()
