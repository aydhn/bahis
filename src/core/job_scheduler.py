"""
job_scheduler.py – APScheduler tabanlı gelişmiş görev zamanlayıcı.

time.sleep(60) gibi basit döngüler profesyonel değildir.
bahis.py bir işletim sistemi gibi davranmalı ve görevleri yönetmelidir.

Görev türleri:
  cron:     11:00 → Fikstürü çek
  interval: 10 dk → Canlı oranları kontrol et
  date:     2026-02-15 19:00 → Derbi başladığında Live Mode'a geç

APScheduler arka planda thread'ler çalıştırır,
ana döngüyü (bahis.py) bloke etmez.
"""
from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Callable

from loguru import logger

try:
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    from apscheduler.triggers.cron import CronTrigger
    from apscheduler.triggers.interval import IntervalTrigger
    from apscheduler.triggers.date import DateTrigger
    from apscheduler.jobstores.memory import MemoryJobStore
    from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_EXECUTED
    APSCHEDULER_OK = True
except ImportError:
    APSCHEDULER_OK = False
    logger.warning("apscheduler yüklü değil – basit zamanlayıcı kullanılacak.")


class JobScheduler:
    """Gelişmiş görev zamanlayıcı – APScheduler ile cron/interval/date.

    Kullanım:
        scheduler = JobScheduler()
        scheduler.add_cron("fikstür_çek", scraper.fetch, hour=11, minute=0)
        scheduler.add_interval("oran_kontrol", odds.check, minutes=10)
        scheduler.add_date("derbi_live", live.start, "2026-02-15 19:00")
        await scheduler.start()
    """

    def __init__(self, timezone: str = "Europe/Istanbul"):
        self._timezone = timezone
        self._jobs: list[dict] = []
        self._fallback_tasks: list[asyncio.Task] = []

        if APSCHEDULER_OK:
            self._scheduler = AsyncIOScheduler(
                jobstores={"default": MemoryJobStore()},
                timezone=timezone,
            )
            self._scheduler.add_listener(self._on_job_event,
                                         EVENT_JOB_EXECUTED | EVENT_JOB_ERROR)
        else:
            self._scheduler = None

        logger.debug(f"JobScheduler başlatıldı (APScheduler={'✓' if APSCHEDULER_OK else '✗'}).")

    # ═══════════════════════════════════════════
    #  CRON – Belirli saatte çalış
    # ═══════════════════════════════════════════
    def add_cron(self, name: str, func: Callable, *,
                 hour: int | str = "*", minute: int | str = 0,
                 day_of_week: str = "mon-sun", **kwargs):
        """Cron görevi ekler.

        Örnek:
            add_cron("fikstür", fetch_fixtures, hour=11, minute=0)
            add_cron("günlük_rapor", send_report, hour=23, minute=30)
        """
        job_info = {
            "name": name, "type": "cron", "func": func,
            "hour": hour, "minute": minute,
        }
        self._jobs.append(job_info)

        if self._scheduler and APSCHEDULER_OK:
            self._scheduler.add_job(
                func, CronTrigger(
                    hour=hour, minute=minute,
                    day_of_week=day_of_week,
                    timezone=self._timezone,
                ),
                id=name, name=name, replace_existing=True,
                **kwargs,
            )
            logger.info(f"[Scheduler] Cron eklendi: {name} @ {hour}:{minute}")
        return self

    # ═══════════════════════════════════════════
    #  INTERVAL – Belirli aralıkta çalış
    # ═══════════════════════════════════════════
    def add_interval(self, name: str, func: Callable, *,
                     seconds: int = 0, minutes: int = 0,
                     hours: int = 0, **kwargs):
        """Interval görevi ekler.

        Örnek:
            add_interval("oran_kontrol", check_odds, minutes=10)
            add_interval("health_check", healthcheck, minutes=5)
        """
        total_seconds = seconds + minutes * 60 + hours * 3600
        job_info = {
            "name": name, "type": "interval", "func": func,
            "interval_seconds": total_seconds,
        }
        self._jobs.append(job_info)

        if self._scheduler and APSCHEDULER_OK:
            self._scheduler.add_job(
                func, IntervalTrigger(
                    seconds=seconds, minutes=minutes, hours=hours,
                ),
                id=name, name=name, replace_existing=True,
                **kwargs,
            )
            logger.info(f"[Scheduler] Interval eklendi: {name} her {total_seconds}s")
        return self

    # ═══════════════════════════════════════════
    #  DATE – Tek seferlik belirli tarihte çalış
    # ═══════════════════════════════════════════
    def add_date(self, name: str, func: Callable, *,
                 run_date: str | datetime, **kwargs):
        """Tek seferlik görev ekler.

        Örnek:
            add_date("derbi_live", switch_to_live, run_date="2026-02-15 19:00")
        """
        if isinstance(run_date, str):
            run_date = datetime.fromisoformat(run_date)

        job_info = {
            "name": name, "type": "date", "func": func,
            "run_date": run_date.isoformat(),
        }
        self._jobs.append(job_info)

        if self._scheduler and APSCHEDULER_OK:
            self._scheduler.add_job(
                func, DateTrigger(run_date=run_date, timezone=self._timezone),
                id=name, name=name, replace_existing=True,
                **kwargs,
            )
            logger.info(f"[Scheduler] Date eklendi: {name} @ {run_date}")
        return self

    # ═══════════════════════════════════════════
    #  BAŞLAT / DURDUR
    # ═══════════════════════════════════════════
    async def start(self):
        """Zamanlayıcıyı başlatır."""
        if self._scheduler and APSCHEDULER_OK:
            self._scheduler.start()
            logger.success(f"[Scheduler] Başlatıldı – {len(self._jobs)} görev.")
        else:
            await self._start_fallback()

    async def stop(self):
        """Zamanlayıcıyı durdurur."""
        if self._scheduler and APSCHEDULER_OK:
            self._scheduler.shutdown(wait=False)
            logger.info("[Scheduler] Durduruldu.")
        else:
            for task in self._fallback_tasks:
                task.cancel()

    # ═══════════════════════════════════════════
    #  FALLBACK: APScheduler yoksa asyncio döngüsü
    # ═══════════════════════════════════════════
    async def _start_fallback(self):
        """APScheduler yokken basit asyncio loop ile görev çalıştır."""
        logger.info("[Scheduler] Fallback modu – asyncio döngüsü.")
        for job in self._jobs:
            if job["type"] == "interval":
                task = asyncio.create_task(
                    self._fallback_interval(
                        job["name"], job["func"], job["interval_seconds"]
                    )
                )
                self._fallback_tasks.append(task)
            elif job["type"] == "date":
                task = asyncio.create_task(
                    self._fallback_date(
                        job["name"], job["func"],
                        datetime.fromisoformat(job["run_date"]),
                    )
                )
                self._fallback_tasks.append(task)

    async def _fallback_interval(self, name: str, func: Callable,
                                  interval: int):
        while True:
            try:
                if asyncio.iscoroutinefunction(func):
                    await func()
                else:
                    func()
            except Exception as e:
                logger.error(f"[Scheduler:fallback] {name} hatası: {e}")
            await asyncio.sleep(interval)

    async def _fallback_date(self, name: str, func: Callable,
                              run_date: datetime):
        now = datetime.now()
        delay = (run_date - now).total_seconds()
        if delay > 0:
            await asyncio.sleep(delay)
        try:
            if asyncio.iscoroutinefunction(func):
                await func()
            else:
                func()
        except Exception as e:
            logger.error(f"[Scheduler:fallback] {name} hatası: {e}")

    # ═══════════════════════════════════════════
    #  EVENT HANDLER
    # ═══════════════════════════════════════════
    @staticmethod
    def _on_job_event(event):
        if event.exception:
            logger.error(f"[Scheduler] Görev hatası: {event.job_id} – {event.exception}")
        else:
            logger.debug(f"[Scheduler] Görev tamamlandı: {event.job_id}")

    # ═══════════════════════════════════════════
    #  DURUM
    # ═══════════════════════════════════════════
    def remove_job(self, name: str):
        if self._scheduler and APSCHEDULER_OK:
            try:
                self._scheduler.remove_job(name)
            except Exception:
                pass

    def get_jobs(self) -> list[dict]:
        """Tüm zamanlanmış görevleri listeler."""
        if self._scheduler and APSCHEDULER_OK:
            return [
                {
                    "id": j.id,
                    "name": j.name,
                    "next_run": str(j.next_run_time) if j.next_run_time else "N/A",
                    "trigger": str(j.trigger),
                }
                for j in self._scheduler.get_jobs()
            ]
        return self._jobs

    @property
    def is_running(self) -> bool:
        if self._scheduler and APSCHEDULER_OK:
            return self._scheduler.running
        return bool(self._fallback_tasks)
