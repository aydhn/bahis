"""
stream_processor.py – Stream Processing (Canlı Veri Akışı).

Polling (veri çekme) yavaştır. Stream processing ile veri
geldiği an tüm modüller eşzamanlı tetiklenir.

Kavramlar:
  - Event Stream: Olay akışı — her oran değişimi bir olay
  - Stream Processing: Olayları gerçek zamanlı işleme
  - Windowed Aggregation: Zaman penceresi tabanlı toplama
    (son 5dk ortalaması vb.)
  - Watermark: Gecikmeli olaylar için zaman damgası yönetimi
  - Backpressure: Tüketici yavaşsa üreticiyi yavaşlatma
  - Fan-out: Bir olayı birden fazla tüketiciye dağıtma
  - Sink: İşlenmiş olayların yazıldığı hedef (DB, Telegram)

Akış:
  1. api_hijacker'dan gelen veri → Stream'e gönderilir
  2. StreamProcessor olayı alır → fan-out ile modüllere dağıtır
  3. Her modül (GARCH, Wavelet, RL) eşzamanlı işler
  4. Sonuçlar birleştirilir (merge) → karar noktasına gider
  5. Karar → Event Bus / Telegram

Teknoloji: Bytewax veya Faust
Fallback: asyncio.Queue + asyncio.Task tabanlı mini-stream
"""
from __future__ import annotations

import asyncio
import importlib.util
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable

from loguru import logger

try:
    import bytewax
    BYTEWAX_OK = True
except ImportError:
    BYTEWAX_OK = False

FAUST_OK = importlib.util.find_spec("faust") is not None

if not BYTEWAX_OK and not FAUST_OK:
    logger.debug("bytewax/faust yüklü değil – asyncio stream fallback.")


# ═══════════════════════════════════════════════
#  VERİ YAPILARI
# ═══════════════════════════════════════════════
@dataclass
class StreamEvent:
    """Akış olayı."""
    event_type: str = ""        # "odds_update" | "goal" | "card" | "lineup"
    match_id: str = ""
    timestamp: float = 0.0
    data: dict = field(default_factory=dict)
    source: str = ""             # "sofascore" | "mackolik" | "manual"


@dataclass
class StreamStats:
    """Akış istatistikleri."""
    total_events: int = 0
    events_per_sec: float = 0.0
    active_consumers: int = 0
    queue_size: int = 0
    avg_latency_ms: float = 0.0
    errors: int = 0
    uptime_sec: float = 0.0


@dataclass
class WindowedAggregate:
    """Pencere tabanlı toplama sonucu."""
    window_start: float = 0.0
    window_end: float = 0.0
    match_id: str = ""
    # İstatistikler
    count: int = 0
    avg_value: float = 0.0
    min_value: float = 0.0
    max_value: float = 0.0
    std_value: float = 0.0
    last_value: float = 0.0


# ═══════════════════════════════════════════════
#  WINDOWED BUFFER
# ═══════════════════════════════════════════════
class WindowedBuffer:
    """Zaman penceresi tabanlı olay tamponu."""

    def __init__(self, window_sec: float = 60.0):
        self._window = window_sec
        self._buffers: dict[str, list[tuple[float, float]]] = defaultdict(list)

    def add(self, match_id: str, value: float,
              timestamp: float | None = None) -> None:
        ts = timestamp or time.time()
        self._buffers[match_id].append((ts, value))
        self._evict(match_id, ts)

    def get_aggregate(self, match_id: str) -> WindowedAggregate | None:
        events = self._buffers.get(match_id, [])
        if not events:
            return None

        values = [v for _, v in events]
        import numpy as np
        agg = WindowedAggregate(
            window_start=events[0][0],
            window_end=events[-1][0],
            match_id=match_id,
            count=len(values),
            avg_value=round(float(np.mean(values)), 6),
            min_value=round(float(np.min(values)), 6),
            max_value=round(float(np.max(values)), 6),
            std_value=round(float(np.std(values)), 6),
            last_value=round(float(values[-1]), 6),
        )
        return agg

    def _evict(self, match_id: str, now: float) -> None:
        cutoff = now - self._window
        self._buffers[match_id] = [
            (ts, v) for ts, v in self._buffers[match_id]
            if ts >= cutoff
        ]

    def get_all_matches(self) -> list[str]:
        return list(self._buffers.keys())


# ═══════════════════════════════════════════════
#  STREAM PROCESSOR (Ana Sınıf)
# ═══════════════════════════════════════════════
class StreamProcessor:
    """Gerçek zamanlı olay akışı işlemci.

    Kullanım:
        sp = StreamProcessor(max_queue=10000)

        # Tüketici kaydet
        sp.register_consumer("garch", garch_handler)
        sp.register_consumer("wavelet", wavelet_handler)
        sp.register_consumer("logger", log_handler)

        # Başlat
        await sp.start()

        # Olay gönder
        await sp.emit(StreamEvent(
            event_type="odds_update",
            match_id="gs_fb",
            data={"home_odds": 1.85, "away_odds": 4.20},
        ))

        # İstatistikler
        stats = sp.get_stats()
    """

    def __init__(self, max_queue: int = 10000,
                 window_sec: float = 60.0,
                 n_workers: int = 4):
        self._max_queue = max_queue
        self._n_workers = n_workers
        self._queue: asyncio.Queue | None = None
        self._consumers: dict[str, Callable] = {}
        self._running = False
        self._tasks: list[asyncio.Task] = []
        self._windowed = WindowedBuffer(window_sec=window_sec)

        # İstatistikler
        self._total_events = 0
        self._errors = 0
        self._start_time = 0.0
        self._latencies: list[float] = []

        logger.debug(
            f"[Stream] Processor başlatıldı: "
            f"queue={max_queue}, workers={n_workers}"
        )

    def register_consumer(self, name: str,
                            handler: Callable[[StreamEvent], Any]) -> None:
        """Tüketici kaydet (fan-out)."""
        self._consumers[name] = handler
        logger.debug(f"[Stream] Consumer '{name}' kaydedildi.")

    def unregister_consumer(self, name: str) -> None:
        """Tüketici çıkar."""
        self._consumers.pop(name, None)

    async def start(self) -> None:
        """Stream processor'ı başlat."""
        if self._running:
            return

        self._queue = asyncio.Queue(maxsize=self._max_queue)
        self._running = True
        self._start_time = time.time()

        for i in range(self._n_workers):
            task = asyncio.create_task(
                self._worker(i), name=f"stream_worker_{i}",
            )
            self._tasks.append(task)

        logger.info(
            f"[Stream] Başlatıldı: {self._n_workers} worker, "
            f"{len(self._consumers)} consumer"
        )

    async def stop(self) -> None:
        """Stream processor'ı durdur."""
        self._running = False
        for task in self._tasks:
            task.cancel()
        self._tasks.clear()
        logger.info("[Stream] Durduruldu.")

    async def emit(self, event: StreamEvent) -> None:
        """Olay gönder."""
        if not self._running or self._queue is None:
            return

        if event.timestamp == 0:
            event.timestamp = time.time()

        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            self._errors += 1
            logger.warning("[Stream] Kuyruk dolu, olay atlandı.")

    async def emit_batch(self, events: list[StreamEvent]) -> None:
        """Toplu olay gönder."""
        for event in events:
            await self.emit(event)

    async def _worker(self, worker_id: int) -> None:
        """İşçi goroutine."""
        while self._running:
            try:
                if self._queue is None:
                    await asyncio.sleep(0.1)
                    continue

                event = await asyncio.wait_for(
                    self._queue.get(), timeout=1.0,
                )
                t0 = time.perf_counter()
                self._total_events += 1

                # Windowed buffer güncelle
                if "odds" in event.data:
                    self._windowed.add(
                        event.match_id,
                        event.data["odds"],
                        event.timestamp,
                    )

                # Fan-out: tüm tüketicilere dağıt
                tasks = []
                for name, handler in self._consumers.items():
                    if asyncio.iscoroutinefunction(handler):
                        tasks.append(handler(event))
                    else:
                        try:
                            handler(event)
                        except Exception as e:
                            self._errors += 1
                            logger.debug(
                                f"[Stream] {name} hatası: {e}"
                            )

                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)

                latency = (time.perf_counter() - t0) * 1000
                self._latencies.append(latency)
                if len(self._latencies) > 1000:
                    self._latencies = self._latencies[-500:]

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._errors += 1
                logger.debug(f"[Stream] Worker {worker_id} hatası: {e}")

    def get_stats(self) -> StreamStats:
        """İstatistikler."""
        uptime = time.time() - self._start_time if self._start_time else 0

        return StreamStats(
            total_events=self._total_events,
            events_per_sec=round(
                self._total_events / max(uptime, 1), 2,
            ),
            active_consumers=len(self._consumers),
            queue_size=self._queue.qsize() if self._queue else 0,
            avg_latency_ms=round(
                float(sum(self._latencies) / max(len(self._latencies), 1)),
                3,
            ),
            errors=self._errors,
            uptime_sec=round(uptime, 1),
        )

    def get_window(self, match_id: str) -> WindowedAggregate | None:
        """Pencere toplama sonucu."""
        return self._windowed.get_aggregate(match_id)

    def get_active_matches(self) -> list[str]:
        """Aktif maçlar."""
        return self._windowed.get_all_matches()
