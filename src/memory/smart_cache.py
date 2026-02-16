"""
smart_cache.py – Akıllı RAM Önbellekleme.

Her analiz için veritabanına gitmek maliyetlidir.
Son 1 saatte çekilen veriler RAM'de tutulur.
Aynı maçı tekrar analiz ederken disk I/O yapılmaz.

3 seviyeli cache:
  L1: functools.lru_cache (fonksiyon düzeyi, ms)
  L2: TTL dict cache (saat bazlı, RAM)
  L3: pickle disk cache (gün bazlı, SSD)
"""
from __future__ import annotations

import functools
import hashlib
import pickle
import time
from pathlib import Path
from typing import Any, Callable

from loguru import logger

CACHE_DIR = Path(__file__).resolve().parents[2] / "cache"


class TTLCache:
    """Time-To-Live RAM cache. Varsayılan: 1 saat TTL.

    Kullanım:
        cache = TTLCache(ttl_seconds=3600)
        cache.set("gs_vs_fb", analysis_data)
        result = cache.get("gs_vs_fb")  # 1 saat içinde → RAM'den
    """

    def __init__(self, ttl_seconds: float = 3600.0, max_size: int = 500):
        self._ttl = ttl_seconds
        self._max_size = max_size
        self._store: dict[str, tuple[float, Any]] = {}
        self._hits = 0
        self._misses = 0
        logger.debug(f"TTLCache başlatıldı: TTL={ttl_seconds}s, max={max_size}")

    def get(self, key: str) -> Any | None:
        """Cache'ten veri al. Süresi dolmuşsa None döner."""
        entry = self._store.get(key)
        if entry is None:
            self._misses += 1
            return None

        ts, value = entry
        if time.time() - ts > self._ttl:
            del self._store[key]
            self._misses += 1
            return None

        self._hits += 1
        return value

    def set(self, key: str, value: Any) -> None:
        """Cache'e veri yaz."""
        if len(self._store) >= self._max_size:
            self._evict_oldest()
        self._store[key] = (time.time(), value)

    def get_or_compute(self, key: str, compute_fn: Callable[[], Any]) -> Any:
        """Cache'te varsa döndür, yoksa hesapla ve kaydet."""
        cached = self.get(key)
        if cached is not None:
            return cached
        result = compute_fn()
        self.set(key, result)
        return result

    def invalidate(self, key: str) -> bool:
        """Tek bir key'i sil."""
        if key in self._store:
            del self._store[key]
            return True
        return False

    def invalidate_pattern(self, pattern: str) -> int:
        """Pattern'e uyan key'leri sil."""
        keys_to_del = [k for k in self._store if pattern in k]
        for k in keys_to_del:
            del self._store[k]
        return len(keys_to_del)

    def clear(self) -> None:
        self._store.clear()

    def _evict_oldest(self):
        """En eski (veya süresi dolmuş) entry'yi sil."""
        now = time.time()
        expired = [k for k, (ts, _) in self._store.items() if now - ts > self._ttl]
        for k in expired:
            del self._store[k]

        if len(self._store) >= self._max_size:
            oldest_key = min(self._store, key=lambda k: self._store[k][0])
            del self._store[oldest_key]

    @property
    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "size": len(self._store),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / max(total, 1),
            "ttl_seconds": self._ttl,
        }

    def __contains__(self, key: str) -> bool:
        return self.get(key) is not None

    def __len__(self) -> int:
        return len(self._store)


class SmartCache:
    """3 seviyeli akıllı cache sistemi.

    L1: LRU (fonksiyon düzeyi) – mikrosaniye
    L2: TTL RAM cache – 1 saat
    L3: Pickle disk – 1 gün
    """

    def __init__(self, ttl_l2: float = 3600.0, ttl_l3: float = 86400.0):
        self._l2 = TTLCache(ttl_seconds=ttl_l2, max_size=1000)
        self._ttl_l3 = ttl_l3
        self._l3_dir = CACHE_DIR / "l3"
        self._l3_dir.mkdir(parents=True, exist_ok=True)
        logger.debug("SmartCache başlatıldı (L1+L2+L3).")

    def get(self, key: str) -> Any | None:
        """L2 → L3 sırasıyla ara."""
        # L2 (RAM)
        l2_val = self._l2.get(key)
        if l2_val is not None:
            return l2_val

        # L3 (disk)
        l3_val = self._l3_get(key)
        if l3_val is not None:
            self._l2.set(key, l3_val)  # L3 → L2 promote
            return l3_val

        return None

    def set(self, key: str, value: Any, persist: bool = True) -> None:
        """L2'ye yaz, persist=True ise L3'e de yaz."""
        self._l2.set(key, value)
        if persist:
            self._l3_set(key, value)

    def get_or_compute(self, key: str, compute_fn: Callable[[], Any],
                       persist: bool = True) -> Any:
        """Cache'te varsa döndür, yoksa hesapla."""
        cached = self.get(key)
        if cached is not None:
            return cached
        result = compute_fn()
        self.set(key, result, persist=persist)
        return result

    # ─── L3 Disk (Pickle) ───
    def _l3_path(self, key: str) -> Path:
        h = hashlib.md5(key.encode()).hexdigest()
        return self._l3_dir / f"{h}.pkl"

    def _l3_get(self, key: str) -> Any | None:
        path = self._l3_path(key)
        if not path.exists():
            return None
        age = time.time() - path.stat().st_mtime
        if age > self._ttl_l3:
            path.unlink(missing_ok=True)
            return None
        try:
            return pickle.loads(path.read_bytes())
        except Exception:
            path.unlink(missing_ok=True)
            return None

    def _l3_set(self, key: str, value: Any):
        path = self._l3_path(key)
        try:
            path.write_bytes(pickle.dumps(value, protocol=5))
        except Exception as e:
            logger.debug(f"L3 cache yazma hatası: {e}")

    def invalidate(self, key: str):
        self._l2.invalidate(key)
        self._l3_path(key).unlink(missing_ok=True)

    def clear_all(self):
        self._l2.clear()
        for f in self._l3_dir.glob("*.pkl"):
            f.unlink(missing_ok=True)

    @property
    def stats(self) -> dict:
        l3_files = list(self._l3_dir.glob("*.pkl"))
        return {
            "l2": self._l2.stats,
            "l3_files": len(l3_files),
            "l3_size_mb": sum(f.stat().st_size for f in l3_files) / (1024 * 1024),
        }


def cached(ttl: float = 3600.0, maxsize: int = 128):
    """Dekoratör: fonksiyon sonucunu LRU + TTL ile cache'le.

    @cached(ttl=3600)
    def expensive_analysis(match_id):
        ...
    """
    def decorator(fn: Callable) -> Callable:
        _cache: dict[str, tuple[float, Any]] = {}

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            key = f"{fn.__name__}:{args}:{sorted(kwargs.items())}"
            entry = _cache.get(key)
            if entry and time.time() - entry[0] < ttl:
                return entry[1]

            result = fn(*args, **kwargs)
            _cache[key] = (time.time(), result)

            # Maxsize aşılırsa en eskiyi sil
            if len(_cache) > maxsize:
                oldest = min(_cache, key=lambda k: _cache[k][0])
                del _cache[oldest]

            return result

        wrapper.cache_clear = _cache.clear
        wrapper.cache_info = lambda: {"size": len(_cache), "ttl": ttl}
        return wrapper

    return decorator
