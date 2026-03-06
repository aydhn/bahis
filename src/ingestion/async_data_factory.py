"""
async_data_factory.py – Playwright + Asyncio tabanlı asenkron veri fabrikası.
Birden fazla kaynaktan eş zamanlı oran ve maç verisi çeker.
"""
from __future__ import annotations

import asyncio
import time
from src.extensions.smart_money import SmartMoneyDetector
import re
from datetime import datetime
from typing import Any

import httpx
import polars as pl
from loguru import logger


class DataFactory:
    """Çoklu kaynaktan eş zamanlı veri toplayan asenkron fabrika."""

    # Ücretsiz / açık API kaynakları
    SOURCES = {
        "the_odds_api": "https://api.the-odds-api.com/v4/sports",
        "football_data": "https://api.football-data.org/v4",
        "flashscore_scrape": "https://www.flashscore.com.tr/",
    }

    def __init__(self, db, cache, headless: bool = True):
        self._db = db
        self._cache = cache
        self._headless = headless
        self._client: httpx.AsyncClient | None = None
        self._browser = None
        logger.debug("DataFactory başlatıldı.")
        self.smart_money = SmartMoneyDetector()

    async def _ensure_client(self):
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=30,
                headers={"User-Agent": "QuantBot/1.0"},
                follow_redirects=True,
            )

    async def _ensure_browser(self):
        if self._browser is None:
            try:
                from playwright.async_api import async_playwright
                pw = await async_playwright().start()
                self._browser = await pw.chromium.launch(headless=self._headless)
                logger.info("Playwright tarayıcısı başlatıldı.")
            except Exception as e:
                logger.warning(f"Playwright başlatılamadı: {e}")



    # ── Scraper: FlashScore ──
    async def _scrape_flashscore(self) -> list[dict]:
        await self._ensure_browser()
        if self._browser is None:
            return []
        matches = []
        try:
            page = await self._browser.new_page()
            await page.goto("https://www.flashscore.com.tr/futbol/turkiye/super-lig/", timeout=30000)
            await page.wait_for_selector(".event__match", timeout=10000)
            elements = await page.query_selector_all(".event__match")
            for el in elements[:50]:
                try:
                    home = await el.query_selector(".event__homeParticipant")
                    away = await el.query_selector(".event__awayParticipant")
                    time_el = await el.query_selector(".event__time")
                    home_name = (await home.inner_text()).strip() if home else ""
                    away_name = (await away.inner_text()).strip() if away else ""
                    match_time = (await time_el.inner_text()).strip() if time_el else ""
                    if home_name and away_name:
                        matches.append({
                            "home_team": home_name,
                            "away_team": away_name,
                            "time": match_time,
                            "source": "flashscore",
                        })
                except Exception:
                    continue
            await page.close()
        except Exception as e:
            logger.warning(f"FlashScore scrape hatası: {e}")
        return matches

    # ── Veriyi normalize et ve DB'ye kaydet ──
    def _normalize_and_store(self, raw_data: list[dict], source: str):
        for item in raw_data:
            match_id = self._generate_match_id(item, source)
            normalized = {
                "match_id": match_id,
                "league": item.get("league", item.get("competition", {}).get("name", "Süper Lig")),
                "home_team": item.get("home_team", item.get("homeTeam", {}).get("name", "")),
                "away_team": item.get("away_team", item.get("awayTeam", {}).get("name", "")),
                "kickoff": self._parse_kickoff(item),
                "status": "upcoming",
            }
            # Oran bilgileri varsa ekle
            odds = self._extract_odds(item)
            normalized.update(odds)

            if "home_odds" in odds:
                steam_signal = self.smart_money.detect_steam(match_id, odds["home_odds"], time.time())
                if steam_signal:
                    logger.info(f"Steam Move Detected for {match_id}: {steam_signal}")

            if normalized["home_team"] and normalized["away_team"]:
                self._db.upsert_match(normalized)

    def _generate_match_id(self, item: dict, source: str) -> str:
        home = item.get("home_team", item.get("homeTeam", {}).get("name", ""))
        away = item.get("away_team", item.get("awayTeam", {}).get("name", ""))
        clean = re.sub(r"[^a-z]", "", f"{home}{away}".lower())
        return f"{clean[:20]}_{source[:4]}"

    def _parse_kickoff(self, item: dict) -> str:
        utc_date = item.get("utcDate", item.get("commence_time", ""))
        if utc_date:
            return utc_date
        return datetime.utcnow().isoformat()

    def _extract_odds(self, item: dict) -> dict:
        odds = {}
        bookmakers = item.get("bookmakers", [])
        if bookmakers:
            bk = bookmakers[0]
            for market in bk.get("markets", []):
                if market.get("key") == "h2h":
                    for outcome in market.get("outcomes", []):
                        if outcome["name"] == "Home" or outcome["name"] == item.get("home_team"):
                            odds["home_odds"] = outcome["price"]
                        elif outcome["name"] == "Away" or outcome["name"] == item.get("away_team"):
                            odds["away_odds"] = outcome["price"]
                        elif outcome["name"] == "Draw":
                            odds["draw_odds"] = outcome["price"]
                elif market.get("key") == "totals":
                    for outcome in market.get("outcomes", []):
                        if outcome["name"] == "Over":
                            odds["over25_odds"] = outcome["price"]
                        elif outcome["name"] == "Under":
                            odds["under25_odds"] = outcome["price"]
        return odds

    # ── Ana çalıştırma döngüleri ──
    async def run_prematch(self, shutdown: asyncio.Event):
        """Pre-match verileri periyodik olarak çeker."""
        logger.info("Pre-match veri fabrikası başlatıldı.")
        while not shutdown.is_set():
            try:
                tasks = [
                    self._scrape_flashscore()
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for i, res in enumerate(results):
                    if isinstance(res, list) and res:
                        src = ["flashscore"][i]
                        self._normalize_and_store(res, src)
                        logger.info(f"{src}: {len(res)} maç çekildi.")
            except Exception as e:
                logger.error(f"Pre-match döngü hatası: {e}")
            await asyncio.sleep(300)  # 5 dakikada bir

    async def run_live(self, shutdown: asyncio.Event):
        """Canlı maç verilerini yakalar (FlashScore scrape)."""
        logger.info("Canlı veri fabrikası başlatıldı.")
        while not shutdown.is_set():
            try:
                matches = await self._scrape_flashscore()
                if matches:
                    self._normalize_and_store(matches, "flashscore")
                    logger.info(f"FlashScore: {len(matches)} maç tarandı.")
            except Exception as e:
                logger.error(f"Canlı döngü hatası: {e}")
            await asyncio.sleep(60)  # 1 dakikada bir

    async def close(self):
        if self._client:
            await self._client.aclose()
        if self._browser:
            await self._browser.close()
