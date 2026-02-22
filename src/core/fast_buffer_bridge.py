"""
rust_bridge.py – Rust FastBuffer modülü için Python Wrapper.

Bu modül, derlenmiş .so / .dll dosyasını yükler ve 
üst seviye Python API'sı sunar.
"""
import ctypes
import os
from loguru import logger

class RustFastBuffer:
    def __init__(self, size: int = 1000, lib_path: str = "src/core/rust_engine/target/release/fast_buffer.dll"):
        self.size = size
        self._lib = None
        self._ptr = None

        if not os.path.exists(lib_path):
            logger.warning(f"Rust kütüphanesi bulunamadı: {lib_path}. Mock modda çalışılıyor.")
            return

        try:
            self._lib = ctypes.CDLL(lib_path)
            
            # Fonksiyon Tanımları
            self._lib.create_buffer.argtypes = [ctypes.c_size_t]
            self._lib.create_buffer.restype = ctypes.c_void_p
            
            self._lib.push_odds.argtypes = [ctypes.c_void_p, ctypes.c_double]
            
            self._lib.get_avg_odds.argtypes = [ctypes.c_void_p]
            self._lib.get_avg_odds.restype = ctypes.c_double
            
            self._lib.destroy_buffer.argtypes = [ctypes.c_void_p]

            # Başlat
            self._ptr = self._lib.create_buffer(size)
            logger.info(f"Rust FastBuffer başlatıldı (size={size})")
        except Exception as e:
            logger.error(f"Rust kütüphanesi yükleme hatası: {e}")

    def push(self, val: float):
        if self._ptr:
            self._lib.push_odds(self._ptr, float(val))

    def get_avg(self) -> float:
        if self._ptr:
            return float(self._lib.get_avg_odds(self._ptr))
        return 0.0

    def __del__(self):
        if self._lib and self._ptr:
            self._lib.destroy_buffer(self._ptr)
            
    def run_batch(self, odds_list: list, **kwargs):
        """Pipeline uyumlu toplu işlem."""
        for o in odds_list:
            self.push(o)
        return self.get_avg()
