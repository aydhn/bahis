"""
metric_exporter.py – Prometheus client ile sistem metrikleri.
CPU, RAM, model latency ve veri pipeline durumunu dışa aktarır.
"""
from __future__ import annotations

import asyncio
import os
import time
from functools import wraps

import psutil
from loguru import logger

try:
    from prometheus_client import (
        Counter, Gauge, Histogram, Summary,
        start_http_server, CollectorRegistry, REGISTRY,
    )
    PROM_AVAILABLE = True
except ImportError:
    PROM_AVAILABLE = False
    logger.warning("prometheus_client yüklü değil – metrikler devre dışı.")


class MetricExporter:
    """Prometheus metrikleri sunucusu."""

    def __init__(self, port: int = 9090):
        self._port = port

        if PROM_AVAILABLE:
            self.cycles_total = Counter("bot_cycles_total", "Toplam analiz döngüsü")
            self.signals_total = Counter("bot_signals_total", "Üretilen toplam sinyal")
            self.errors_total = Counter("bot_errors_total", "Toplam hata sayısı")
            self.model_latency = Histogram(
                "bot_model_latency_seconds", "Model tahmin süresi",
                buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
            )
            self.cpu_percent = Gauge("bot_cpu_percent", "CPU kullanımı %")
            self.memory_mb = Gauge("bot_memory_mb", "RAM kullanımı MB")
            self.active_tasks = Gauge("bot_active_tasks", "Aktif asyncio görevleri")
            self.data_freshness = Gauge("bot_data_freshness_sec", "Verinin yaşı (saniye)")
        else:
            self.cycles_total = _NoopMetric()
            self.signals_total = _NoopMetric()
            self.errors_total = _NoopMetric()
            self.model_latency = _NoopMetric()
            self.cpu_percent = _NoopMetric()
            self.memory_mb = _NoopMetric()
            self.active_tasks = _NoopMetric()
            self.data_freshness = _NoopMetric()

        logger.debug("MetricExporter başlatıldı.")

    async def serve(self):
        """Prometheus HTTP sunucusunu başlatır ve periyodik sistem metriklerini günceller."""
        if PROM_AVAILABLE:
            try:
                start_http_server(self._port)
                logger.info(f"Prometheus metrikleri → http://localhost:{self._port}/metrics")
            except OSError as e:
                logger.warning(f"Prometheus port hatası: {e}")

        while True:
            self._update_system_metrics()
            await asyncio.sleep(5)

    def _update_system_metrics(self):
        proc = psutil.Process(os.getpid())
        if PROM_AVAILABLE:
            self.cpu_percent.set(proc.cpu_percent())
            self.memory_mb.set(proc.memory_info().rss / 1e6)
            self.active_tasks.set(len(asyncio.all_tasks()) if asyncio.get_event_loop().is_running() else 0)

    def track_latency(self, fn):
        """Dekoratör: fonksiyon süresini ölçer."""
        @wraps(fn)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = fn(*args, **kwargs)
            elapsed = time.perf_counter() - start
            if PROM_AVAILABLE:
                self.model_latency.observe(elapsed)
            return result
        return wrapper


class _NoopMetric:
    """Prometheus yokken sessiz çalışan sahte metrik."""
    def inc(self, *a, **kw): pass
    def set(self, *a, **kw): pass
    def observe(self, *a, **kw): pass
    def labels(self, *a, **kw): return self
    def time(self): return _NoopTimer()


class _NoopTimer:
    def __enter__(self): return self
    def __exit__(self, *a): pass
