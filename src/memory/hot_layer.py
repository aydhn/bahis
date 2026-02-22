"""
hot_layer.py – RAM-Accelerated Hot-Layer (Shared Memory / O(1) Cache).

Sistemin en sık eriştiği verileri (canlı oranlar, aktif sinyaller, 
kritik feature setleri) RAM üzerinde tutar. 
DuckDB disk I/O maliyetini sıfıra indirerek milisaniyelik gecikme hedefler.
"""
from __future__ import annotations
import time
from typing import Dict, Any, Optional
from loguru import logger
import threading

class HotLayer:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(HotLayer, cls).__new__(cls)
        return cls._instance

    def __init__(self, ttl_sec: int = 3600):
        if not hasattr(self, "_initialized"):
            self._storage: Dict[str, Any] = {}
            self._expirations: Dict[str, float] = {}
            self._ttl = ttl_sec
            self._lock = threading.Lock()
            self._initialized = True
            logger.info("[HotLayer] RAM-Accelerated katmanı aktif.")

    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Veriyi RAM'e yazar."""
        with self._lock:
            self._storage[key] = value
            self._expirations[key] = time.time() + (ttl or self._ttl)

    def get(self, key: str) -> Any:
        """Veriyi RAM'den okur. O(1) karmaşıklık."""
        with self._lock:
            if key not in self._storage:
                return None
            
            if time.time() > self._expirations.get(key, 0):
                self._delete(key)
                return None
                
            return self._storage[key]

    def _delete(self, key: str):
        if key in self._storage:
            del self._storage[key]
        if key in self._expirations:
            del self._expirations[key]

    def flush(self):
        """Tüm RAM önbelleğini temizler."""
        with self._lock:
            self._storage.clear()
            self._expirations.clear()
            logger.debug("[HotLayer] Cache temizlendi.")

    async def run_batch(self, **kwargs):
        """Pipeline entegrasyonu: Bayat verileri temizle."""
        now = time.time()
        with self._lock:
            expired_keys = [k for k, exp in self._expirations.items() if now > exp]
            for k in expired_keys:
                self._delete(k)
        if expired_keys:
            logger.debug(f"[HotLayer] {len(expired_keys)} bayat kayıt temizlendi.")

# Global singleton instance
hot_layer = HotLayer()
