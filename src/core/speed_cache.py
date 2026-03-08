"""
speed_cache.py – High-Performance In-Memory Data Store.

Hot data (live odds, signals, ticks) requires faster access than SQLite.
SpeedCache uses Python's collections.deque and dict for O(1) access.
Thread-safe for concurrent pipeline stages.
"""
from __future__ import annotations

import time
import threading
import itertools
from collections import deque
from dataclasses import dataclass
from typing import Any, Optional, Dict, List

from loguru import logger

@dataclass
class CacheEntry:
    value: Any
    timestamp: float
    ttl: float = 0.0  # 0 = infinite

class SpeedCache:
    """
    High-Performance In-Memory Store.

    Features:
    - O(1) Key-Value Access
    - O(1) FIFO Queue for Streams (Odds Ticks)
    - Thread-Safe (Locks)
    - TTL Support
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(SpeedCache, cls).__new__(cls)
                    cls._instance._init()
        return cls._instance

    def _init(self):
        self._kv_store: Dict[str, CacheEntry] = {}
        self._streams: Dict[str, deque] = {}
        self._stream_limits: Dict[str, int] = {}
        self._rw_lock = threading.RLock()
        logger.debug("SpeedCache initialized (Singleton).")

    def set(self, key: str, value: Any, ttl: float = 0.0):
        """Set a key-value pair with optional TTL (seconds)."""
        with self._rw_lock:
            self._kv_store[key] = CacheEntry(
                value=value,
                timestamp=time.time(),
                ttl=ttl
            )

    def get(self, key: str) -> Optional[Any]:
        """Get a value. Returns None if expired or missing."""
        with self._rw_lock:
            entry = self._kv_store.get(key)
            if not entry:
                return None

            if entry.ttl > 0 and (time.time() - entry.timestamp > entry.ttl):
                del self._kv_store[key]
                return None

            return entry.value

    def push_stream(self, stream_name: str, item: Any, max_len: int = 1000):
        """Push item to a named stream (FIFO)."""
        with self._rw_lock:
            if stream_name not in self._streams:
                self._streams[stream_name] = deque(maxlen=max_len)
                self._stream_limits[stream_name] = max_len

            # Update maxlen if changed
            if self._stream_limits[stream_name] != max_len:
                # deque doesn't support dynamic resizing easily, we re-create if needed or just accept current
                # Ideally, create new deque with new maxlen and extend
                old_deque = self._streams[stream_name]
                if old_deque.maxlen != max_len:
                    self._streams[stream_name] = deque(old_deque, maxlen=max_len)
                    self._stream_limits[stream_name] = max_len

            self._streams[stream_name].append({
                "data": item,
                "ts": time.time()
            })

    def get_stream(self, stream_name: str, limit: int = 10) -> List[Any]:
        """Get latest N items from stream."""
        with self._rw_lock:
            stream = self._streams.get(stream_name)
            if not stream:
                return []

            # Return last N items
            # Optimization: Use islice on reversed stream to avoid copying entire deque
            # O(k) instead of O(N) where k=limit, N=stream length
            items = list(itertools.islice(reversed(stream), limit))
            return items[::-1]

    def clear(self):
        """Clear all data."""
        with self._rw_lock:
            self._kv_store.clear()
            self._streams.clear()
            logger.debug("SpeedCache cleared.")

    def stats(self) -> Dict[str, int]:
        """Return basic stats."""
        with self._rw_lock:
            return {
                "keys": len(self._kv_store),
                "streams": len(self._streams),
                "total_stream_items": sum(len(s) for s in self._streams.values())
            }
