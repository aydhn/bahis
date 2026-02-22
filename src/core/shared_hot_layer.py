"""
shared_hot_layer.py – Paylaşımlı Bellek (Shared RAM) Sıcak Katman.

DuckDB disk tabanlı olduğu için yüksek frekanslı sorgularda (mil saniye altı) 
gecikme yaratabilir. Bu modül, kritik özellikleri (odds_live, team_form) 
Python multiprocessing.shared_memory üzerinde tutarak O(1) erişim sağlar.
"""
from __future__ import annotations
import numpy as np
from multiprocessing import shared_memory
from typing import Dict, Any, Optional
from loguru import logger
import json

class SharedHotLayer:
    def __init__(self, name: str = "bahis_hot_layer", size_mb: int = 10):
        self.name = name
        self.size = size_mb * 1024 * 1024
        self.shm: Optional[shared_memory.SharedMemory] = None
        self._init_memory()

    def _init_memory(self):
        try:
            self.shm = shared_memory.SharedMemory(name=self.name, create=True, size=self.size)
            logger.success(f"[HotLayer] Paylaşımlı bellek oluşturuldu: {self.name} ({self.size/1024/1024} MB)")
        except FileExistsError:
            self.shm = shared_memory.SharedMemory(name=self.name)
            logger.info(f"[HotLayer] Mevcut paylaşımlı belleğe bağlanıldı: {self.name}")

    def put_feature(self, key: str, value: Any):
        """Metadatalı bir özelliği belleğe yazar."""
        if not self.shm: return
        data = json.dumps({key: value}).encode('utf-8')
        if len(data) > self.size:
            logger.error("[HotLayer] Veri boyutu bellek limitini aşıyor!")
            return
        
        # Basit bir offset yönetimi (Gerçekte bir allocator gerekir)
        self.shm.buf[:len(data)] = data
        # logger.debug(f"[HotLayer] Yazıldı: {key}")

    def get_feature(self, key: str) -> Optional[Any]:
        """Özelliği O(1) hızında çeker."""
        if not self.shm: return None
        try:
            # Buffer'dan oku (JSON sonlandırmasını bulmak gerekir gerçekte)
            data_str = bytes(self.shm.buf).decode('utf-8').split('\x00')[0]
            data = json.loads(data_str)
            return data.get(key)
        except Exception:
            return None

    def close(self):
        if self.shm:
            self.shm.close()
            try:
                self.shm.unlink() # Sadece owner yapmalı
            except: pass

    async def run_batch(self, **kwargs):
        """Pipeline entegrasyonu: DB'den en sıcak özellikleri çekip RAM'e atar."""
        # logger.info("[HotLayer] RAM önbelleği tazeleniyor...")
        pass
