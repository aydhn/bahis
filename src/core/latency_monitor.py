"""
latency_monitor.py – Sistem genelinde gecikme (latency) takibi ve optimizasyonu.

Veri toplama, analiz ve işlem süreçlerindeki gecikmeleri nanosaniye 
hassasiyetinde ölçer ve darboğazları raporlar.
"""
import time
from loguru import logger
from collections import deque
from typing import Dict, List

class UltraLatencyMonitor:
    def __init__(self, window_size: int = 100):
        self._history = deque(maxlen=window_size)
        self._checkpoints: Dict[str, float] = {}

    def start_timer(self, label: str):
        """Bir işlem için zamanlayıcıyı başlatır."""
        self._checkpoints[label] = time.perf_counter_ns()

    def stop_timer(self, label: str) -> float:
        """Zamanlayıcıyı durdurur ve geçen süreyi (ms) döner."""
        if label not in self._checkpoints:
            return 0.0
        
        elapsed_ns = time.perf_counter_ns() - self._checkpoints.pop(label)
        elapsed_ms = elapsed_ns / 1_000_000.0
        self._history.append(elapsed_ms)
        
        if elapsed_ms > 100.0: # 100ms üzeri gecikmeler kritik kabul edilir
            logger.warning(f"[Latency] Yüksek gecikme tespit edildi: {label} -> {elapsed_ms:.2f}ms")
            
        return elapsed_ms

    def get_stats(self) -> Dict[str, float]:
        """Gecikme istatistiklerini döner."""
        if not self._history:
            return {"avg_ms": 0.0, "p99_ms": 0.0}
            
        data = sorted(self._history)
        return {
            "avg_ms": round(sum(data) / len(data), 4),
            "p99_ms": round(data[int(len(data) * 0.99)], 4),
            "max_ms": round(max(data), 4)
        }
