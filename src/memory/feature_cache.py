"""
feature_cache.py – DiskCache tabanlı SSD önbellek sistemi.
Ağır feature hesaplamalarını cache'ler, TTL ile eskiyen veriyi temizler.
"""
from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Any, Callable

from diskcache import Cache
from loguru import logger

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache"


class FeatureCache:
    """SSD üzerinde disk-tabanlı önbellek (Memoization)."""

    def __init__(self, directory: Path | str = CACHE_DIR, size_limit_gb: float = 2.0):
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._cache = Cache(str(self._dir), size_limit=int(size_limit_gb * 1e9))
        logger.debug(f"FeatureCache başlatıldı → {self._dir} (limit {size_limit_gb} GB)")

    def _make_key(self, name: str, *args, **kwargs) -> str:
        raw = f"{name}:{args}:{sorted(kwargs.items())}"
        return hashlib.sha256(raw.encode()).hexdigest()[:24]

    def get_or_compute(self, key: str, compute_fn: Callable, ttl: int = 300) -> Any:
        """Cache'te varsa döndür, yoksa hesapla, kaydet, döndür."""
        cached = self._cache.get(key)
        if cached is not None:
            logger.debug(f"Cache HIT: {key}")
            return cached
        logger.debug(f"Cache MISS: {key} – hesaplanıyor…")
        result = compute_fn()
        self._cache.set(key, result, expire=ttl)
        return result

    def memoize(self, ttl: int = 300):
        """Dekoratör: fonksiyon sonuçlarını cache'ler."""
        def decorator(fn: Callable):
            def wrapper(*args, **kwargs):
                key = self._make_key(fn.__name__, *args, **kwargs)
                return self.get_or_compute(key, lambda: fn(*args, **kwargs), ttl=ttl)
            wrapper.__name__ = fn.__name__
            wrapper.__doc__ = fn.__doc__
            return wrapper
        return decorator

    def invalidate(self, key: str):
        self._cache.delete(key)

    def clear_all(self):
        self._cache.clear()
        logger.warning("Tüm cache temizlendi.")

    def stats(self) -> dict:
        return {
            "size_bytes": self._cache.volume(),
            "count": len(self._cache),
            "directory": str(self._dir),
        }
