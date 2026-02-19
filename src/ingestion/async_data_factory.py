"""
async_data_factory.py – Playwright + Asyncio tabanlı asenkron veri fabrikası.
Birden fazla kaynaktan eş zamanlı oran ve maç verisi çeker.
"""
from __future__ import annotations

import asyncio
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

    # Football-data.org ücretsiz ligler (TR1 YOK - ücretsiz tier kapsamında değil)
    FOOTBALL_DATA_COMPETITIONS = {
        "PL": "Premier League",
        "BL1": "Bundesliga",
        "SA": "Serie A",
        "PD": "La Liga",
        "FL1": "Ligue 1",
        "CL": "Champions League",
        # "EC" ve "WC" sezon dışında 404 döndürüyor - kaldırıldı
    }

    # Sofascore Türk ligleri için ID'ler
    SOFASCORE_TOURNAMENT_IDS = {
        "super_lig": 52,        # Süper Lig
        "tff1": 98,             # TFF 1. Lig
        "tff2": 3024,           # TFF 2. Lig
        "turkish_cup": 231,     # Türkiye Kupası
    }

    def __init__(self, db, cache, headless: bool = True):
        self._db = db
        self._cache = cache
        self._headless = headless
        self._client: httpx.AsyncClient | None = None
        self._browser = None
        self._pw = None  # Playwright instance - tek tutulur
        self._browser_lock = asyncio.Lock()
        self._browser_fail_count = 0
        self._browser_max_fails = 5
        logger.debug("DataFactory başlatıldı.")

    async def _ensure_client(self):
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=30,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                    "Accept-Language": "tr-TR,tr;q=0.9",
                },
                follow_redirects=True,
                verify=False,  # SSL sertifika sorunlarını aşmak için
            )

    async def _ensure_browser(self):
        """Browser sağlıklı değilse yeniden başlat."""
        async with self._browser_lock:
            if self._browser_fail_count >= self._browser_max_fails:
                logger.warning(f"[Browser] {self._browser_fail_count} başarısız denemeden sonra browser devre dışı.")
                return

            browser_healthy = False
            if self._browser is not None:
                try:
                    # Browser hâlâ çalışıyor mu kontrol et
                    _ = self._browser.contexts
                    browser_healthy = True
                except Exception:
                    browser_healthy = False

            if not browser_healthy:
                # Eski kaynakları temizle
                try:
                    if self._browser:
                        await self._browser.close()
                except Exception:
                    pass
                try:
                    if self._pw:
                        await self._pw.stop()
                except Exception:
                    pass
                self._browser = None
                self._pw = None

                try:
                    from playwright.async_api import async_playwright
                    self._pw = await async_playwright().start()
                    self._browser = await self._pw.chromium.launch(
                        headless=True,
                        args=[
                            "--no-sandbox",
                            "--disable-setuid-sandbox",
                            "--disable-dev-shm-usage",
                            "--disable-gpu",
                        ],
                    )
                    logger.info("Playwright tarayıcısı (yeniden) başlatıldı.")
                except Exception as e:
                    self._browser_fail_count += 1
                    logger.warning(f"Playwright başlatılamadı ({self._browser_fail_count}/{self._browser_max_fails}): {e}")

    # ── API: The Odds API (ücretsiz tier) ──
    async def _fetch_odds_api(self, api_key: str = "") -> list[dict]:
        await self._ensure_client()
        if not api_key:
            logger.debug("Odds API key yok – atlanıyor.")
            return []
        url = f"{self.SOURCES['the_odds_api']}/soccer_turkey_super_league/odds"
        params = {
            "apiKey": api_key,
            "regions": "eu",
            "markets": "h2h,totals",
            "oddsFormat": "decimal",
        }
        try:
            resp = await self._client.get(url, params=params)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.warning(f"Odds API hatası: {e}")
            return []

    # ── API: Football-Data.org (ücretsiz tier – TR1 hariç) ──
    async def _fetch_football_data(self, api_key: str = "") -> list[dict]:
        await self._ensure_client()
        if not api_key:
            logger.debug("[FD.org] API key yok – Sofascore fallback'e geçiliyor.")
            return await self._fetch_sofascore_scheduled()

        headers = {"X-Auth-Token": api_key}
        all_matches = []
        for comp_code in self.FOOTBALL_DATA_COMPETITIONS:
            try:
                resp = await self._client.get(
                    f"{self.SOURCES['football_data']}/competitions/{comp_code}/matches",
                    headers=headers,
                    params={"status": "SCHEDULED"},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    matches = data.get("matches", [])
                    for m in matches:
                        m["_competition_code"] = comp_code
                        m["sport"] = "football"
                    all_matches.extend(matches)
                elif resp.status_code == 429:
                    logger.debug("[FD.org] Rate limit – durduruluyor.")
                    await asyncio.sleep(10)
                    break
                elif resp.status_code == 404:
                    logger.debug(f"[FD.org] {comp_code} bulunamadı (404) – atlanıyor.")
                else:
                    logger.debug(f"[FD.org] {comp_code}: HTTP {resp.status_code}")
            except Exception as e:
                logger.debug(f"[FD.org] {comp_code} hatası: {e}")
        if all_matches:
            logger.info(f"[FD.org] {len(all_matches)} maç çekildi.")
        return all_matches

    # ── Sofascore API fallback (ücretsiz, no-auth) ──
    async def _fetch_sofascore_scheduled(self) -> list[dict]:
        """Sofascore'dan Türk ligi maçlarını çek."""
        await self._ensure_client()
        from datetime import datetime as _dt
        today = _dt.now().strftime("%Y-%m-%d")
        results = []
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
                "Accept": "application/json",
                "Referer": "https://www.sofascore.com/",
                "Origin": "https://www.sofascore.com",
            }
            resp = await self._client.get(
                f"https://api.sofascore.com/api/v1/sport/football/scheduled-events/{today}",
                headers=headers,
                timeout=15,
            )
            if resp.status_code == 200:
                events = resp.json().get("events", [])
                for ev in events:
                    home = ev.get("homeTeam", {}).get("name", "")
                    away = ev.get("awayTeam", {}).get("name", "")
                    if home and away:
                        results.append({
                            "match_id": f"ss_{ev.get('id', '')}",
                            "home_team": home,
                            "away_team": away,
                            "league": ev.get("tournament", {}).get("name", ""),
                            "status": "upcoming",
                            "sport": "football",
                        })
                logger.info(f"[Sofascore-fallback] {len(results)} maç çekildi.")
        except Exception as e:
            logger.debug(f"[Sofascore-fallback] {e}")
        return results

    # ── Scraper: FlashScore (güvenli browser ile) ──
    async def _scrape_flashscore(self) -> list[dict]:
        await self._ensure_browser()
        if self._browser is None:
            return []
        matches = []
        page = None
        try:
            ctx = await self._browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                viewport={"width": 1280, "height": 720},
            )
            page = await ctx.new_page()
            page.set_default_timeout(20000)

            await page.goto(
                "https://www.flashscore.com.tr/futbol/turkiye/super-lig/",
                wait_until="domcontentloaded",
                timeout=25000,
            )
            # Daha güvenilir selector bekle
            try:
                await page.wait_for_selector(".event__match, .sportName--soccer", timeout=8000)
            except Exception:
                pass  # Selector bulunamazsa yine de devam et

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
                            "status": "upcoming",
                        })
                except Exception:
                    continue

            logger.info(f"FlashScore: {len(matches)} maç tarandı.")
        except Exception as e:
            logger.warning(f"FlashScore scrape hatası: {e}")
            # Browser sorunluysa sıfırla
            self._browser = None
        finally:
            if page is not None:
                try:
                    await page.close()
                except Exception:
                    pass
            try:
                await ctx.close()  # type: ignore
            except Exception:
                pass
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
                    self._fetch_odds_api(),
                    self._fetch_football_data(),
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for i, res in enumerate(results):
                    if isinstance(res, list) and res:
                        src = ["odds_api", "football_data"][i]
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
