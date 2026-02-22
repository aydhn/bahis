"""
scraper_agent.py – Sofascore, Mackolik, Transfermarkt scraper ajanları.
Selenium headless + BeautifulSoup4 ile veri çeker.
IP ban'den korunmak için fake-useragent + rastgele jitter.
"""
from __future__ import annotations

import asyncio
import random
import re
import time
from datetime import datetime
from typing import Any

from loguru import logger
from src.ingestion.proxy_manager import ProxyManager

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    logger.warning("beautifulsoup4 yüklü değil – scraper devre dışı.")

try:
    from fake_useragent import UserAgent
    _ua = UserAgent()
    def random_ua() -> str:
        return _ua.random
except ImportError:
    def random_ua() -> str:
        agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 Safari/605.1.15",
            "Mozilla/5.0 (X11; Linux x86_64; rv:121.0) Gecko/20100101 Firefox/121.0",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Edge/120.0.0.0",
        ]
        return random.choice(agents)

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False


def _jitter(min_s: float = 1.0, max_s: float = 4.0):
    """Rastgele bekleme süresi – bot tespitinden kaçınmak için."""
    time.sleep(random.uniform(min_s, max_s))


async def _async_jitter(min_s: float = 1.0, max_s: float = 4.0):
    await asyncio.sleep(random.uniform(min_s, max_s))


class BaseScraper:
    """Tüm scraper'ların temel sınıfı."""

    def __init__(self, name: str, proxy_mgr: ProxyManager = None):
        self.name = name
        self._client: httpx.AsyncClient | None = None
        self._session_count = 0
        self._proxy_mgr = proxy_mgr
        self._current_proxy = None

    async def _ensure_client(self):
        if self._client is None or self._client.is_closed:
            proxies = None
            if self._proxy_mgr:
                p = self._proxy_mgr.get_proxy()
                if p:
                    self._current_proxy = p
                    proxies = {"http://": p, "https://": p}
                    logger.info(f"[{self.name}] Using Proxy: {p}")

            self._client = httpx.AsyncClient(
                timeout=30,
                headers={
                    "User-Agent": random_ua(),
                    "Accept-Language": "tr-TR,tr;q=0.9,en;q=0.8",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                },
                follow_redirects=True,
                proxies=proxies,
                verify=False, # Often needed for free proxies
            )

    async def rotate_session(self):
        """Force session rotation (close old, clear proxy, re-init)."""
        logger.warning(f"[{self.name}] Rotating Session & Proxy...")
        if self._client:
            await self._client.aclose()
        
        # Mark current proxy as bad if it failed specifically (e.g. 403)
        if self._proxy_mgr and self._current_proxy:
            self._proxy_mgr.mark_bad(self._current_proxy)
            self._current_proxy = None
            
        self._client = None
        # Next call to _ensure_client will pick a new proxy


    async def _fetch(self, url: str) -> str | None:
        """URL'den HTML çeker, jitter ile."""
        await self._ensure_client()
        await _async_jitter(1.5, 4.0)

        # Her 10 istekte User-Agent değiştir
        self._session_count += 1
        if self._session_count % 10 == 0:
            self._client.headers["User-Agent"] = random_ua()

        try:
            resp = await self._client.get(url)
            resp.raise_for_status()
            logger.debug(f"[{self.name}] {url[:60]}… → {resp.status_code}")
            return resp.text
        except Exception as e:
            logger.warning(f"[{self.name}] Fetch hatası ({url[:50]}): {e}")
            return None

    async def close(self):
        if self._client:
            await self._client.aclose()


# ═══════════════════════════════════════════════════════
#  MACKOLIK SCRAPER
# ═══════════════════════════════════════════════════════
class MackolikScraper(BaseScraper):
    """Mackolik.com'dan maç ve oran verisi çeker.

    Fallback zinciri:
      1. mackolik.com/canli-sonuclar  (fixture)
      2. mackolik.com/iddaa           (oran – yeni URL; /iddaa/tum-maclar 404)
      3. mackolik.com/iddaa/bugunku-maclar (alternatif)
      4. Tüm URL'ler başarısız → boş liste dön
    """

    BASE_URL = "https://www.mackolik.com"
    FIXTURE_URLS = [
        "/canli-sonuclar",
        "/futbol/canli-sonuclar",
        "/iddaa/program",
    ]
    ODDS_URLS = [
        "/iddaa/bulten",
        "/iddaa/program",
    ]

    def __init__(self, proxy_mgr: ProxyManager = None):
        super().__init__("Mackolik", proxy_mgr)

    async def _fetch_first_ok(self, url_paths: list[str]) -> str | None:
        """URL listesini sırayla dene, ilk 200 yanıtı dön."""
        for path in url_paths:
            url = self.BASE_URL + path
            html = await self._fetch(url)
            if html:
                # 404 sayfası kontrolü: içeriğin çok kısa olmadığını doğrula
                if len(html) > 2000:
                    logger.debug(f"[Mackolik] Çalışan URL: {url}")
                    return html
                logger.debug(f"[Mackolik] {url} → boş/hatalı içerik ({len(html)}b)")
            else:
                logger.debug(f"[Mackolik] {url} → başarısız")
        logger.warning("[Mackolik] Tüm URL'ler başarısız – Mackolik atlanıyor.")
        return None

    async def scrape_fixtures(self) -> list[dict]:
        """Günün maçlarını çeker (fallback URL zinciri ile)."""
        html = await self._fetch_first_ok(self.FIXTURE_URLS)
        if not html or not BS4_AVAILABLE:
            return []

        soup = BeautifulSoup(html, "lxml")
        matches = []

        rows = soup.select(
            ".match-row, .p-match, [data-type='match'], "
            ".match-item, .fixture-row, [class*='match']"
        )
        for row in rows[:50]:
            try:
                home_el = row.select_one(
                    ".home-team, .team-home, .p-home, [class*='home']"
                )
                away_el = row.select_one(
                    ".away-team, .team-away, .p-away, [class*='away']"
                )
                time_el = row.select_one(".match-time, .p-time, .time, [class*='time']")

                home = home_el.get_text(strip=True) if home_el else ""
                away = away_el.get_text(strip=True) if away_el else ""
                match_time = time_el.get_text(strip=True) if time_el else ""

                if home and away and len(home) > 1:
                    matches.append({
                        "home_team": home,
                        "away_team": away,
                        "time": match_time,
                        "source": "mackolik",
                        "league": self._extract_league(row),
                    })
            except Exception:
                continue

        logger.info(f"[Mackolik] {len(matches)} maç çekildi.")
        return matches

    async def scrape_odds(self) -> list[dict]:
        """İddaa oranlarını çeker (fallback URL zinciri ile)."""
        html = await self._fetch_first_ok(self.ODDS_URLS)
        if not html or not BS4_AVAILABLE:
            logger.info("[Mackolik] Oran sayfası alınamadı – atlanıyor.")
            return []

        soup = BeautifulSoup(html, "lxml")
        odds_data = []

        rows = soup.select(
            "tr.odd-row, tr[data-match], .iddaa-row, "
            "[class*='iddaa-match'], [class*='bet-row'], "
            "tr[class*='match']"
        )
        for row in rows[:100]:
            try:
                teams_el = row.select_one(
                    ".teams, .match-name, [class*='teams'], [class*='match-name']"
                )
                odds_els = row.select(
                    ".odd-value, .odds, td.odd, [class*='odd'], [class*='oran']"
                )

                if teams_el and len(odds_els) >= 3:
                    teams = teams_el.get_text(strip=True)
                    parts = re.split(r'\s*[-–vs]\s*', teams, maxsplit=1)
                    if len(parts) == 2:
                        odds_data.append({
                            "home_team": parts[0].strip(),
                            "away_team": parts[1].strip(),
                            "home_odds": self._parse_odd(odds_els[0].get_text(strip=True)),
                            "draw_odds": self._parse_odd(odds_els[1].get_text(strip=True)),
                            "away_odds": self._parse_odd(odds_els[2].get_text(strip=True)),
                            "source": "mackolik_iddaa",
                        })
            except Exception:
                continue

        logger.info(f"[Mackolik] {len(odds_data)} oran çekildi.")
        return odds_data

    def _extract_league(self, element) -> str:
        league_el = element.find_previous(".league-name, .p-league, .competition")
        return league_el.get_text(strip=True) if league_el else "Bilinmeyen Lig"

    @staticmethod
    def _parse_odd(text: str) -> float:
        try:
            return float(text.replace(",", ".").strip())
        except (ValueError, AttributeError):
            return 0.0


# ═══════════════════════════════════════════════════════
#  SOFASCORE SCRAPER
# ═══════════════════════════════════════════════════════
class SofascoreScraper(BaseScraper):
    """Sofascore API'den çok sporlu maç verileri çeker.

    Desteklenen sporlar: futbol, basketbol, tenis, voleybol, hentbol, hokey.
    """

    API_BASE = "https://api.sofascore.com/api/v1"
    SPORTS = ["football", "basketball", "tennis", "volleyball", "handball", "ice-hockey"]

    def __init__(self, proxy_mgr: ProxyManager = None):
        super().__init__("Sofascore", proxy_mgr)

    async def scrape_live(self, sport: str = "football") -> list[dict]:
        """Belirli sporun canlı maçlarını API'den çeker – 403 retry ile."""
        await self._ensure_client()
        await _async_jitter(0.5, 2.0)

        _ss_headers = {
            "User-Agent": random_ua(),
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7",
            "Accept-Encoding": "gzip, deflate, br, zstd",
            "Origin": "https://www.sofascore.com",
            "Referer": "https://www.sofascore.com/",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-site",
            "Cache-Control": "max-age=0",
            "Connection": "keep-alive",
            "Priority": "u=1, i",
        }

        endpoints = [
            f"{self.API_BASE}/sport/{sport}/events/live",
        ]
        from datetime import datetime as _dt
        today = _dt.now().strftime("%Y-%m-%d")
        endpoints.append(f"{self.API_BASE}/sport/{sport}/scheduled-events/{today}")

        for attempt, url in enumerate(endpoints):
            try:
                _ss_headers["User-Agent"] = random_ua()  # Rotate UA per attempt
                resp = await self._client.get(url, headers=_ss_headers)
                if resp.status_code == 200:
                    data = resp.json()
                    events = data.get("events", [])
                    matches = []
                    for ev in events[:100]:
                        home = ev.get("homeTeam", {}).get("name", "")
                        away = ev.get("awayTeam", {}).get("name", "")
                        tournament = ev.get("tournament", {})
                        tournament_name = tournament.get("name", "")
                        country = tournament.get("category", {}).get("name", "")
                        home_score = ev.get("homeScore", {}).get("current", 0)
                        away_score = ev.get("awayScore", {}).get("current", 0)
                        matches.append({
                            "match_id": f"sofa_{ev.get('id', '')}",
                            "home_team": home,
                            "away_team": away,
                            "league": tournament_name,
                            "country": country,
                            "sport": sport,
                            "home_score": home_score,
                            "away_score": away_score,
                            "status": "live",
                            "source": "sofascore",
                        })
                    logger.info(f"[Sofascore] {sport}: {len(matches)} maç çekildi.")
                    return matches
                elif resp.status_code in (403, 429):
                    logger.warning(f"[{self.name}] {sport} API yanıtı: {resp.status_code} -> Rotating Proxy...")
                    await self.rotate_session()     
                    await self._ensure_client() # Re-init immediately
                    await _async_jitter(1.0, 3.0)  # Backoff before next attempt
                    # Retry logic handles next iteration if designed (the loop continues to next endpoint usually)
                    # Ideally we want to retry THE SAME endpoint with new proxy.
                    # But for now let's just rotate and let next endpoint attempt utilize new proxy.
                    continue
                else:
                    logger.debug(f"[Sofascore] {sport} API yanıtı: {resp.status_code}")
            except Exception as e:
                logger.warning(f"[Sofascore] {sport} API hatası: {e}")

        logger.debug(f"[Sofascore] {sport}: tüm endpoint'ler başarısız.")
        return []

    async def scrape_all_live(self) -> list[dict]:
        """TÜM sporlardan canlı maçları çek."""
        all_matches = []
        for sport in self.SPORTS:
            try:
                matches = await self.scrape_live(sport)
                all_matches.extend(matches)
            except Exception as e:
                logger.debug(f"[Sofascore] {sport} canlı hata: {e}")
        return all_matches

    async def scrape_team_stats(self, team_id: int) -> dict:
        """Takım istatistiklerini çeker."""
        await _async_jitter(2.0, 5.0)
        try:
            resp = await self._client.get(
                f"{self.API_BASE}/team/{team_id}/statistics/overall",
                headers={"User-Agent": random_ua()},
            )
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            logger.debug(f"[Sofascore] Takım stat hatası: {e}")
        return {}

    async def scrape_match_stats(self, event_id: int) -> dict:
        """Maç detaylı istatistiklerini çeker."""
        await _async_jitter(1.5, 3.5)
        try:
            resp = await self._client.get(
                f"{self.API_BASE}/event/{event_id}/statistics",
                headers={"User-Agent": random_ua()},
            )
            if resp.status_code == 200:
                data = resp.json()
                stats = {}
                for period in data.get("statistics", []):
                    for group in period.get("groups", []):
                        for item in group.get("statisticsItems", []):
                            stats[item["name"]] = {
                                "home": item.get("home", ""),
                                "away": item.get("away", ""),
                            }
                return stats
        except Exception as e:
            logger.debug(f"[Sofascore] Maç stat hatası: {e}")
        return {}


# ═══════════════════════════════════════════════════════
#  TRANSFERMARKT SCRAPER
# ═══════════════════════════════════════════════════════
class TransfermarktScraper(BaseScraper):
    """Transfermarkt'tan piyasa değeri ve kadro bilgisi çeker."""

    BASE_URL = "https://www.transfermarkt.com.tr"

    def __init__(self, proxy_mgr: ProxyManager = None):
        super().__init__("Transfermarkt", proxy_mgr)

    async def scrape_squad_value(self, team_slug: str) -> dict:
        """Takım kadro piyasa değerini çeker."""
        url = f"{self.BASE_URL}/{team_slug}/kader/verein/"
        html = await self._fetch(url)
        if not html or not BS4_AVAILABLE:
            return {}

        soup = BeautifulSoup(html, "lxml")
        result = {"team": team_slug, "players": [], "total_value": 0}

        rows = soup.select("table.items tbody tr")
        for row in rows[:30]:
            try:
                name_el = row.select_one(".hauptlink a")
                value_el = row.select_one(".rechts.hauptlink")
                pos_el = row.select_one("td:nth-child(2)")

                if name_el:
                    player = {
                        "name": name_el.get_text(strip=True),
                        "position": pos_el.get_text(strip=True) if pos_el else "",
                        "market_value": self._parse_value(value_el.get_text(strip=True)) if value_el else 0,
                    }
                    result["players"].append(player)
                    result["total_value"] += player["market_value"]
            except Exception:
                continue

        logger.info(f"[Transfermarkt] {team_slug}: {len(result['players'])} oyuncu çekildi.")
        return result

    async def scrape_injuries(self, team_slug: str) -> list[dict]:
        """Takım sakatlık listesini çeker."""
        url = f"{self.BASE_URL}/{team_slug}/ausfaelle/verein/"
        html = await self._fetch(url)
        if not html or not BS4_AVAILABLE:
            return []

        soup = BeautifulSoup(html, "lxml")
        injuries = []

        rows = soup.select("table.items tbody tr")
        for row in rows[:20]:
            try:
                name_el = row.select_one(".hauptlink a")
                reason_el = row.select_one(".verletzungsgrund")
                since_el = row.select_one("td:nth-child(3)")

                if name_el:
                    injuries.append({
                        "player": name_el.get_text(strip=True),
                        "reason": reason_el.get_text(strip=True) if reason_el else "Bilinmeyen",
                        "since": since_el.get_text(strip=True) if since_el else "",
                    })
            except Exception:
                continue

        logger.info(f"[Transfermarkt] {team_slug}: {len(injuries)} sakatlık çekildi.")
        return injuries

    @staticmethod
    def _parse_value(text: str) -> float:
        """'25,00 mil. €' → 25000000.0"""
        text = text.replace(".", "").replace(",", ".").strip()
        multiplier = 1
        if "mil" in text.lower():
            multiplier = 1_000_000
        elif "bin" in text.lower():
            multiplier = 1_000
        numbers = re.findall(r"[\d.]+", text)
        if numbers:
            return float(numbers[0]) * multiplier
        return 0.0


# ═══════════════════════════════════════════════════════
#  ANA SCRAPER AJAN
# ═══════════════════════════════════════════════════════
class ScraperAgent:
    """Tüm scraper'ları yöneten ana ajan – Circuit Breaker korumalı.

    Senaryo: Mackolik IP engeli yerse 3 hatada devre açılır,
    1 saat boyunca çağrılmaz. Bu sürede Sofascore devam eder.
    """

    def __init__(self, db, notifier=None, cb_registry=None):
        self._db = db
        self._notifier = notifier
        
        # Initialize Proxy Manager
        self._proxy_mgr = ProxyManager() # Uses FREE_LIST by default
        
        self._mackolik = MackolikScraper(self._proxy_mgr)
        self._sofascore = SofascoreScraper(self._proxy_mgr)
        self._transfermarkt = TransfermarktScraper(self._proxy_mgr)

        # Self-Healing Engine (Otopoyetik Kurtarma)
        from src.core.auto_healer import SelfHealingEngine
        self._healer = SelfHealingEngine(llm_backend="auto")

        # Her scraper'a ayrı circuit breaker
        from src.core.circuit_breaker import CircuitBreakerRegistry, CBConfig
        self._cb_registry = cb_registry or CircuitBreakerRegistry()
        scraper_config = CBConfig(
            failure_threshold=3,
            recovery_timeout=3600.0,
            half_open_max_calls=2,
            success_threshold=2,
            backoff_multiplier=1.5,
            max_recovery_timeout=14400.0,
        )
        self._cb_mackolik = self._cb_registry.get_or_create("scraper:mackolik", "scraper")
        self._cb_sofascore = self._cb_registry.get_or_create("scraper:sofascore", "scraper")
        self._cb_transfermarkt = self._cb_registry.get_or_create("scraper:transfermarkt", "scraper")

        # Devre açıldığında Telegram'a bildir
        if self._notifier:
            self._cb_mackolik.on_open(self._notify_open)
            self._cb_sofascore.on_open(self._notify_open)
            self._cb_transfermarkt.on_open(self._notify_open)

        logger.debug("ScraperAgent başlatıldı (Self-Healing + CB aktif).")

    def _notify_open(self, name: str, error: str):
        """Circuit breaker açıldığında bildirim kuyruğuna ekle."""
        if self._notifier:
            asyncio.ensure_future(
                self._notifier.send_scraper_down(name, error)
            )

    async def run_all(self, shutdown: asyncio.Event):
        """Tüm scraper'ları periyodik olarak çalıştırır."""
        logger.info("ScraperAgent – tüm ajanlar başlatılıyor…")
        while not shutdown.is_set():
            tasks = []

            # Sadece devresi kapalı olan scraper'ları çalıştır
            if self._cb_mackolik.is_available:
                tasks.append(self._scrape_mackolik())
            else:
                remaining = self._cb_mackolik.time_until_recovery
                logger.info(
                    f"[CB] Mackolik devre AÇIK – atlanıyor "
                    f"(kalan: {remaining/60:.0f} dk)"
                )

            if self._cb_sofascore.is_available:
                tasks.append(self._scrape_sofascore())
            else:
                remaining = self._cb_sofascore.time_until_recovery
                logger.info(f"[CB] Sofascore devre AÇIK – atlanıyor (kalan: {remaining/60:.0f} dk)")

            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            else:
                logger.warning("[CB] TÜM scraper'lar devre AÇIK – bekleniyor…")

            await asyncio.sleep(180)

    async def _scrape_mackolik(self):
        async def _do():
            try:
                fixtures = await self._mackolik.scrape_fixtures()
                odds = await self._mackolik.scrape_odds()
                self._store_matches(fixtures, "mackolik")
                self._store_odds(odds)
                return len(fixtures) + len(odds)
            except Exception as e:
                # KRİTİK: Hata durumunda kendini iyileştirme motorunu tetikle
                logger.error(f"[ScraperAgent] Mackolik hatası: {e}. İyileştirme tetikleniyor...")
                await self._healer.attempt_heal(e, module_path="src/ingestion/scraper_agent.py")
                raise e

        result = await self._cb_mackolik.call_async(_do)
        if result:
            logger.debug(f"[CB:mackolik] Başarılı – {result} kayıt.")

    async def _scrape_sofascore(self):
        async def _do():
            try:
                all_matches = await self._sofascore.scrape_all_live()
                self._store_matches(all_matches, "sofascore")
                return len(all_matches)
            except Exception as e:
                logger.error(f"[ScraperAgent] Sofascore hatası: {e}. İyileştirme tetikleniyor...")
                await self._healer.attempt_heal(e, module_path="src/ingestion/scraper_agent.py")
                raise e

        result = await self._cb_sofascore.call_async(_do)
        if result:
            logger.debug(f"[CB:sofascore] Başarılı – {result} kayıt (çok sporlu).")

    def _store_matches(self, matches: list[dict], source: str):
        for m in matches:
            match_id = m.get("match_id", "")
            if not match_id:
                home = m.get("home_team", "")
                away = m.get("away_team", "")
                clean = re.sub(r"[^a-z]", "", f"{home}{away}".lower())[:20]
                match_id = f"{clean}_{source[:4]}"

            self._db.upsert_match({
                "match_id": match_id,
                "sport": m.get("sport", "football"),
                "league": m.get("league", ""),
                "country": m.get("country", ""),
                "home_team": m.get("home_team", ""),
                "away_team": m.get("away_team", ""),
                "kickoff": m.get("kickoff", datetime.utcnow().isoformat()),
                "status": m.get("status", "upcoming"),
                "home_odds": m.get("home_odds", None),
                "draw_odds": m.get("draw_odds", None),
                "away_odds": m.get("away_odds", None),
            })

    def _store_odds(self, odds: list[dict]):
        for o in odds:
            home = o.get("home_team", "")
            away = o.get("away_team", "")
            clean = re.sub(r"[^a-z]", "", f"{home}{away}".lower())[:20]
            match_id = f"{clean}_idda"

            self._db.upsert_match({
                "match_id": match_id,
                "home_team": home,
                "away_team": away,
                "home_odds": o.get("home_odds"),
                "draw_odds": o.get("draw_odds"),
                "away_odds": o.get("away_odds"),
                "status": "upcoming",
            })

            if o.get("home_odds"):
                self._db.insert_odds_tick(match_id, "iddaa", "1X2", "home", o["home_odds"])
            if o.get("draw_odds"):
                self._db.insert_odds_tick(match_id, "iddaa", "1X2", "draw", o["draw_odds"])
            if o.get("away_odds"):
                self._db.insert_odds_tick(match_id, "iddaa", "1X2", "away", o["away_odds"])

    async def close(self):
        await self._mackolik.close()
        await self._sofascore.close()
        await self._transfermarkt.close()
