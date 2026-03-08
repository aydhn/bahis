"""
event_bus.py – Event Sourcing & Replayability (Zaman Yolculuğu).

Veritabanına sadece son durumu kaydetmeyin.
Olayları kaydedin:
  Maç Başladı → Gol (dk 10) → Oran Değişti (1.50 → 1.80)

Bu sayede sistemi geçmişteki herhangi bir ana geri döndürüp
(Rollback) bugünkü kodunuzla tekrar çalıştırabilirsiniz.

"Geçen haftaki derbide sistemim neden yanlış karar verdi?"
→ O maçı tekrar oynat → Bugünkü kodla test et.

Teknoloji:
  - SQLite (events.db) ile kalıcı olay deposu
  - In-memory pub/sub event bus
  - Replay Engine (geçmiş olayları yeniden oynatma)
"""
from __future__ import annotations

import asyncio
import json
import sqlite3
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable

from loguru import logger

ROOT = Path(__file__).resolve().parent.parent.parent
DB_PATH = ROOT / "data" / "events.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════
#  EVENT MODEL
# ═══════════════════════════════════════════════
@dataclass
class Event:
    """Tek bir sistem olayı."""
    event_id: str = ""
    event_type: str = ""          # match_started | goal | odds_changed | bet_placed | ...
    source: str = ""              # scraper | model | portfolio | telegram | ...
    match_id: str = ""
    timestamp: float = 0.0
    data: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    cycle: int = 0

    def __post_init__(self):
        if not self.event_id:
            self.event_id = str(uuid.uuid4())[:12]
        if not self.timestamp:
            self.timestamp = time.time()

    def to_dict(self) -> dict:
        return asdict(self)


# ═══════════════════════════════════════════════
#  EVENT STORE (SQLite)
# ═══════════════════════════════════════════════
class EventStore:
    """SQLite tabanlı kalıcı olay deposu.

    Tüm olaylar kronolojik sırada saklanır.
    Zaman aralığına göre sorgulama ve replay yapılabilir.
    """

    def __init__(self, db_path: str | Path | None = None):
        self._path = str(db_path or DB_PATH)
        self._conn: sqlite3.Connection | None = None
        self._init_db()

    def _init_db(self):
        """Veritabanı şemasını oluştur."""
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS events (
                event_id    TEXT PRIMARY KEY,
                event_type  TEXT NOT NULL,
                source      TEXT DEFAULT '',
                match_id    TEXT DEFAULT '',
                timestamp   REAL NOT NULL,
                data        TEXT DEFAULT '{}',
                metadata    TEXT DEFAULT '{}',
                cycle       INTEGER DEFAULT 0,
                created_at  TEXT DEFAULT (datetime('now'))
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_events_type
            ON events(event_type)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_events_match
            ON events(match_id)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_events_ts
            ON events(timestamp)
        """)
        self._conn.commit()
        logger.debug(f"[EventStore] Bağlandı: {self._path}")

    def append(self, event: Event):
        """Olayı depoya ekle."""
        if not self._conn:
            return
        try:
            self._conn.execute(
                """INSERT OR IGNORE INTO events
                   (event_id, event_type, source, match_id,
                    timestamp, data, metadata, cycle)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    event.event_id, event.event_type, event.source,
                    event.match_id, event.timestamp,
                    json.dumps(event.data, ensure_ascii=False, default=str),
                    json.dumps(event.metadata, ensure_ascii=False, default=str),
                    event.cycle,
                ),
            )
            self._conn.commit()
        except Exception as e:
            logger.debug(f"[EventStore] Yazma hatası: {e}")

    def append_batch(self, events: list[Event]):
        """Toplu olay ekleme."""
        if not self._conn or not events:
            return
        rows = [
            (e.event_id, e.event_type, e.source, e.match_id,
             e.timestamp,
             json.dumps(e.data, ensure_ascii=False, default=str),
             json.dumps(e.metadata, ensure_ascii=False, default=str),
             e.cycle)
            for e in events
        ]
        try:
            self._conn.executemany(
                """INSERT OR IGNORE INTO events
                   (event_id, event_type, source, match_id,
                    timestamp, data, metadata, cycle)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )
            self._conn.commit()
        except Exception as e:
            logger.debug(f"[EventStore] Toplu yazma hatası: {e}")

    def query(self, event_type: str | None = None,
              match_id: str | None = None,
              start_ts: float | None = None,
              end_ts: float | None = None,
              limit: int = 1000) -> list[Event]:
        """Olayları filtrele."""
        if not self._conn:
            return []

        where = []
        params: list = []

        if event_type:
            where.append("event_type = ?")
            params.append(event_type)
        if match_id:
            where.append("match_id = ?")
            params.append(match_id)
        if start_ts is not None:
            where.append("timestamp >= ?")
            params.append(start_ts)
        if end_ts is not None:
            where.append("timestamp <= ?")
            params.append(end_ts)

        clause = " AND ".join(where) if where else "1=1"
        params.append(limit)

        rows = self._conn.execute(
            f"""SELECT event_id, event_type, source, match_id,
                       timestamp, data, metadata, cycle
                FROM events
                WHERE {clause}
                ORDER BY timestamp ASC
                LIMIT ?""",
            params,
        ).fetchall()

        events = []
        for r in rows:
            events.append(Event(
                event_id=r[0], event_type=r[1], source=r[2],
                match_id=r[3], timestamp=r[4],
                data=json.loads(r[5]) if r[5] else {},
                metadata=json.loads(r[6]) if r[6] else {},
                cycle=r[7],
            ))
        return events

    def count(self, event_type: str | None = None) -> int:
        if not self._conn:
            return 0
        if event_type:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM events WHERE event_type = ?",
                (event_type,),
            ).fetchone()
        else:
            row = self._conn.execute("SELECT COUNT(*) FROM events").fetchone()
        return row[0] if row else 0

    def get_match_timeline(self, match_id: str) -> list[Event]:
        """Bir maçın tüm olay zaman çizelgesi."""
        return self.query(match_id=match_id, limit=10000)

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None


# ═══════════════════════════════════════════════
#  EVENT BUS (Pub/Sub)
# ═══════════════════════════════════════════════
class EventBus:
    """Merkezi olay veriyolu – tüm ajanlar buradan haberleşir.

    Kullanım:
        bus = EventBus()
        # Dinleyici kaydet
        bus.subscribe("goal", my_handler)
        bus.subscribe("odds_changed", my_other_handler)
        # Olay yayınla
        await bus.emit(Event(event_type="goal", match_id="GS_FB", data={"minute": 10}))
    """

    def __init__(self, store: EventStore | None = None,
                 persist: bool = True):
        self._store = store or (EventStore() if persist else None)
        self._subscribers: dict[str, list[Callable]] = {}
        self._global_subscribers: list[Callable] = []
        self._event_count = 0
        self._paused = False
        logger.debug("[EventBus] Başlatıldı.")

    def subscribe(self, event_type: str, handler: Callable):
        """Belirli bir olay tipine abone ol."""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(handler)
        logger.debug(f"[EventBus] '{event_type}' → {handler.__name__}")

    def subscribe_all(self, handler: Callable):
        """Tüm olaylara abone ol."""
        self._global_subscribers.append(handler)

    def unsubscribe(self, event_type: str, handler: Callable):
        """Aboneliği iptal et."""
        if event_type in self._subscribers:
            self._subscribers[event_type] = [
                h for h in self._subscribers[event_type] if h != handler
            ]

    async def emit(self, event: Event):
        """Olay yayınla (tüm abonelere ilet + depoya kaydet)."""
        if self._paused:
            return

        self._event_count += 1

        # Depoya kaydet
        if self._store:
            self._store.append(event)

        # Tip bazlı aboneler
        handlers = self._subscribers.get(event.event_type, [])
        for handler in handlers:
            try:
                result = handler(event)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.debug(
                    f"[EventBus] Handler hatası ({handler.__name__}): {e}"
                )

        # Global aboneler
        for handler in self._global_subscribers:
            try:
                result = handler(event)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                pass

    def emit_sync(self, event: Event):
        """Senkron olay yayını (async olmayan bağlamlarda)."""
        if self._paused:
            return
        self._event_count += 1
        if self._store:
            self._store.append(event)

        handlers = self._subscribers.get(event.event_type, [])
        for handler in handlers:
            try:
                handler(event)
            except Exception:
                pass

    def pause(self):
        """Olay akışını duraklat."""
        self._paused = True

    def resume(self):
        """Olay akışını devam ettir."""
        self._paused = False

    @property
    def event_count(self) -> int:
        return self._event_count

    @property
    def store(self) -> EventStore | None:
        return self._store


# ═══════════════════════════════════════════════
#  REPLAY ENGINE
# ═══════════════════════════════════════════════
class ReplayEngine:
    """Geçmiş olayları yeniden oynatma motoru.

    Kullanım:
        replayer = ReplayEngine(bus)
        # Geçmiş olayları al
        events = bus.store.get_match_timeline("GS_FB")
        # Bugünkü kodla tekrar oynat
        results = await replayer.replay(events, speed=10.0)
    """

    def __init__(self, bus: EventBus):
        self._bus = bus
        self._replay_results: list[dict] = []
        logger.debug("[ReplayEngine] Hazır.")

    async def replay(self, events: list[Event],
                     speed: float = 1.0,
                     real_time: bool = False) -> dict:
        """Olay listesini kronolojik sırada yeniden oynat.

        Args:
            events: Oynatılacak olaylar
            speed: Hız çarpanı (10.0 = 10x hızlı)
            real_time: True ise olay zaman damgalarına göre bekle
        """
        if not events:
            return {"status": "empty", "events_replayed": 0}

        logger.info(
            f"[Replay] {len(events)} olay oynatılıyor (speed={speed}x)"
        )

        events_sorted = sorted(events, key=lambda e: e.timestamp)
        replayed = 0
        errors = 0
        start = time.time()

        for i, event in enumerate(events_sorted):
            # Zaman aralığı simülasyonu
            if real_time and i > 0:
                dt = event.timestamp - events_sorted[i - 1].timestamp
                if dt > 0:
                    await asyncio.sleep(dt / speed)

            # Olayı yeniden yayınla (metadata'ya replay işareti)
            replay_event = Event(
                event_id=f"replay_{event.event_id}",
                event_type=event.event_type,
                source=f"replay:{event.source}",
                match_id=event.match_id,
                timestamp=event.timestamp,
                data=event.data,
                metadata={
                    **event.metadata,
                    "is_replay": True,
                    "original_id": event.event_id,
                },
                cycle=event.cycle,
            )

            try:
                await self._bus.emit(replay_event)
                replayed += 1
            except Exception as e:
                errors += 1
                logger.debug(f"[Replay] Hata (#{i}): {e}")

        elapsed = time.time() - start

        result = {
            "status": "completed",
            "events_replayed": replayed,
            "errors": errors,
            "elapsed_seconds": round(elapsed, 2),
            "speed": speed,
            "time_span": (
                events_sorted[-1].timestamp - events_sorted[0].timestamp
                if len(events_sorted) > 1 else 0
            ),
        }

        logger.success(
            f"[Replay] Tamamlandı: {replayed} olay, "
            f"{elapsed:.1f}s ({errors} hata)"
        )
        return result

    async def replay_match(self, match_id: str,
                           speed: float = 10.0) -> dict:
        """Tek maçın tüm olaylarını yeniden oynat."""
        if not self._bus.store:
            return {"status": "no_store"}

        events = self._bus.store.get_match_timeline(match_id)
        if not events:
            return {"status": "no_events", "match_id": match_id}

        return await self.replay(events, speed=speed, real_time=True)

    async def replay_time_range(self, start_ts: float, end_ts: float,
                                 speed: float = 10.0) -> dict:
        """Zaman aralığındaki tüm olayları oynat."""
        if not self._bus.store:
            return {"status": "no_store"}

        events = self._bus.store.query(start_ts=start_ts, end_ts=end_ts)
        return await self.replay(events, speed=speed, real_time=True)


# ═══════════════════════════════════════════════
#  HELPER: Hızlı olay oluşturma
# ═══════════════════════════════════════════════
def match_event(event_type: str, match_id: str,
                source: str = "system", **data) -> Event:
    """Hızlı maç olayı oluştur."""
    return Event(
        event_type=event_type,
        source=source,
        match_id=match_id,
        data=data,
    )


# Yaygın olay tipleri
EVENT_TYPES = {
    "match_started": "Maç başladı",
    "match_ended": "Maç bitti",
    "goal": "Gol atıldı",
    "red_card": "Kırmızı kart",
    "yellow_card": "Sarı kart",
    "odds_changed": "Oran değişti",
    "odds_opened": "Oran açıldı",
    "bet_placed": "Bahis yapıldı",
    "bet_settled": "Bahis sonuçlandı",
    "signal_generated": "Sinyal üretildi",
    "model_prediction": "Model tahmini",
    "anomaly_detected": "Anomali tespit edildi",
    "scraper_error": "Scraper hatası",
    "system_error": "Sistem hatası",
    "lineup_announced": "Kadro açıklandı",
    "hedge_opportunity": "Hedge fırsatı",
    "abstain": "Bahis pas geçildi",
    "blacklisted": "Kara listeye alındı",
    "shadow_bet": "Gölge bahis",
}
