"""
zero_copy_bridge.py – SharedMemory / mmap tabanlı kopyalamasız veri iletimi.
Büyük numpy dizilerini süreçler arası sıfır-kopya ile paylaşır.
"""
from __future__ import annotations

import numpy as np
from multiprocessing import shared_memory
from loguru import logger


class ZeroCopyBridge:
    """Süreçler arası sıfır-kopya veri paylaşım köprüsü."""

    def __init__(self):
        self._blocks: dict[str, shared_memory.SharedMemory] = {}
        logger.debug("ZeroCopyBridge başlatıldı.")

    def publish(self, name: str, array: np.ndarray) -> dict:
        """Numpy dizisini shared memory'ye yazar. Meta veri döndürür."""
        if name in self._blocks:
            self._blocks[name].close()
            try:
                self._blocks[name].unlink()
            except FileNotFoundError:
                pass

        shm = shared_memory.SharedMemory(create=True, size=array.nbytes)
        buf = np.ndarray(array.shape, dtype=array.dtype, buffer=shm.buf)
        np.copyto(buf, array)
        self._blocks[name] = shm

        meta = {
            "name": shm.name,
            "shape": array.shape,
            "dtype": str(array.dtype),
            "nbytes": array.nbytes,
        }
        logger.debug(f"Published '{name}' → {meta}")
        return meta

    def subscribe(self, meta: dict) -> np.ndarray:
        """Meta veri ile shared memory'den numpy dizisini okur (kopyalamasız)."""
        shm = shared_memory.SharedMemory(name=meta["name"])
        arr = np.ndarray(
            shape=meta["shape"],
            dtype=np.dtype(meta["dtype"]),
            buffer=shm.buf,
        )
        return arr

    def cleanup(self):
        """Tüm shared memory bloklarını temizler."""
        for name, shm in self._blocks.items():
            try:
                shm.close()
                shm.unlink()
            except Exception as e:
                logger.debug(f"Exception caught: {e}")
        self._blocks.clear()
        logger.info("ZeroCopyBridge temizlendi.")

    def __del__(self):
        self.cleanup()
