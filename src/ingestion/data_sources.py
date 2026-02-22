"""
data_sources.py â€“ Ãœcretsiz Ham ve Temiz Veri KaynaklarÄ± ("Gizli Hazineler").

Resmi API'lar (Opta, Sportradar) aylÄ±k binlerce dolar ister.
Bir Quant Developer arka kapÄ±larÄ± bilir:

1. understat       â†’ xG verileri (dÃ¼nyadaki en iyi Ã¼cretsiz kaynak)
2. soccerdata/FBref â†’ DetaylÄ± maÃ§ istatistikleri (ÅŸut, pas, koÅŸu mesafesi)
3. football-csv     â†’ TÃ¼m liglerin tarihsel sonuÃ§larÄ± (CSV)
4. Hidden APIs      â†’ Sofascore/Flashscore JSON endpoint'leri

TÃ¼m kaynaklar â†’ Polars DataFrame â†’ DuckDB veritabanÄ±na akÄ±ÅŸ.
"""
from __future__ import annotations

import asyncio
import io
import os
import time
from datetime import datetime
from typing import Any

from loguru import logger
from src.ingestion.layered_providers import LayeredFreeProviderSystem
from src.ingestion.mock_generator import MockGenerator

try:
    import polars as pl
    POLARS_OK = True
except ImportError:
    POLARS_OK = False

try:
    import httpx
    HTTPX_OK = True
except ImportError:
    HTTPX_OK = False


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  1. UNDERSTAT â€“ xG VERÄ°LERÄ°
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
class UnderstatSource:
    """understat.com â€“ xG verileri (pip install understat).

    DÃ¼nyadaki en kapsamlÄ± Ã¼cretsiz xG kaynaÄŸÄ±.
    Lig: EPL, La_Liga, Bundesliga, Serie_A, Ligue_1, RFPL
    """

    SUPPORTED_LEAGUES = {
        "epl": "EPL",
        "la_liga": "La_liga",
        "bundesliga": "Bundesliga",
        "serie_a": "Serie_A",
        "ligue_1": "Ligue_1",
        "rfpl": "RFPL",
    }

    def __init__(self):
        self._client = None
        self._cache: dict[str, tuple[float, Any]] = {}
        self._cache_ttl = 3600.0

    async def get_league_matches(self, league: str = "epl",
                                  season: str = "2025") -> list[dict]:
        """Lig maÃ§larÄ±nÄ± xG verileriyle Ã§ek."""
        cache_key = f"understat_{league}_{season}"
        cached = self._check_cache(cache_key)
        if cached is not None:
            return cached

        try:
            import understat
            async with understat.UnderstatClient() as client:
                league_name = self.SUPPORTED_LEAGUES.get(league, league)
                matches = await client.get_league_results(league_name, int(season))

                result = []
                for m in matches:
                    result.append({
                        "source": "understat",
                        "match_id": f"us_{m.get('id', '')}",
                        "home_team": m.get("h", {}).get("title", ""),
                        "away_team": m.get("a", {}).get("title", ""),
                        "home_goals": int(m.get("goals", {}).get("h", 0)),
                        "away_goals": int(m.get("goals", {}).get("a", 0)),
                        "home_xg": float(m.get("xG", {}).get("h", 0)),
                        "away_xg": float(m.get("xG", {}).get("a", 0)),
                        "date": m.get("datetime", ""),
                        "league": league,
                        "season": season,
                    })

                self._set_cache(cache_key, result)
                logger.info(f"[Understat] {league}/{season}: {len(result)} maÃ§ Ã§ekildi.")
                return result
        except ImportError:
            logger.warning("[Understat] understat kÃ¼tÃ¼phanesi yÃ¼klÃ¼ deÄŸil: pip install understat")
            return []
        except Exception as e:
            logger.error(f"[Understat] Hata: {e}")
            return []

    async def get_team_xg(self, team: str, league: str = "epl",
                           season: str = "2025") -> list[dict]:
        """Belirli bir takÄ±mÄ±n maÃ§ bazlÄ± xG verileri."""
        try:
            import understat
            async with understat.UnderstatClient() as client:
                players = await client.get_team_stats(team, int(season))
                return players if isinstance(players, list) else []
        except Exception as e:
            logger.debug(f"[Understat] TakÄ±m xG hatasÄ±: {e}")
            return []

    def _check_cache(self, key: str):
        entry = self._cache.get(key)
        if entry and (time.time() - entry[0]) < self._cache_ttl:
            return entry[1]
        return None

    def _set_cache(self, key: str, data: Any):
        self._cache[key] = (time.time(), data)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  2. SOCCERDATA / FBREF â€“ DETAYLI Ä°STATÄ°STÄ°K
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
class FBrefSource:
    """FBref (via soccerdata) â€“ DetaylÄ± maÃ§ istatistikleri.

    ÅŸut, pas, koÅŸu mesafesi, possession, corners...
    pip install soccerdata
    """

    LEAGUE_MAP = {
        "super_lig": "TUR-SÃ¼per Lig",
        "epl": "ENG-Premier League",
        "la_liga": "ESP-La Liga",
        "bundesliga": "GER-Bundesliga",
        "serie_a": "ITA-Serie A",
    }

    def __init__(self):
        self._ws = None

    def get_match_stats(self, league: str = "epl",
                         season: str = "2526") -> list[dict]:
        """FBref'ten detaylÄ± maÃ§ istatistikleri Ã§ek."""
        try:
            import soccerdata as sd
            fbref = sd.FBref(leagues=self.LEAGUE_MAP.get(league, league),
                             seasons=season)

            schedule = fbref.read_schedule()
            if schedule is None or schedule.empty:
                return []

            results = []
            for _, row in schedule.iterrows():
                results.append({
                    "source": "fbref",
                    "match_id": f"fbref_{hash(str(row.name)) % 99999:05d}",
                    "home_team": str(row.get("home_team", "")),
                    "away_team": str(row.get("away_team", "")),
                    "home_goals": int(row.get("home_score", 0)) if row.get("home_score") else 0,
                    "away_goals": int(row.get("away_score", 0)) if row.get("away_score") else 0,
                    "date": str(row.get("date", "")),
                    "league": league,
                })

            logger.info(f"[FBref] {league}/{season}: {len(results)} maÃ§.")
            return results
        except ImportError:
            logger.warning("[FBref] soccerdata yÃ¼klÃ¼ deÄŸil: pip install soccerdata")
            return []
        except Exception as e:
            logger.error(f"[FBref] Hata: {e}")
            return []

    def get_shooting_stats(self, league: str = "epl",
                            season: str = "2526") -> list[dict]:
        """Åut istatistikleri (xG, npxG, key passes)."""
        try:
            import soccerdata as sd
            fbref = sd.FBref(leagues=self.LEAGUE_MAP.get(league, league),
                             seasons=season)
            shooting = fbref.read_team_season_stats(stat_type="shooting")
            if shooting is None or shooting.empty:
                return []

            return shooting.reset_index().to_dict("records")
        except Exception as e:
            logger.debug(f"[FBref] Shooting stats hatasÄ±: {e}")
            return []


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  3. FOOTBALL-CSV â€“ TARÄ°HSEL SONUÃ‡LAR
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
class FootballCSVSource:
    """football-csv (GitHub) â€“ TÃ¼m liglerin CSV sonuÃ§larÄ±.

    DoÄŸrudan URL'den okuma â€“ API kurulumu gerekmez.
    """

    BASE_URL = "https://raw.githubusercontent.com/footballcsv/footballcsv.github.io/master"

    # PopÃ¼ler lig dosya yollarÄ± (Ã¶rnekler)
    LEAGUE_URLS = {
        "epl": "https://raw.githubusercontent.com/openfootball/england/master/2025-26/1-premierleague.csv",
        "super_lig": "https://raw.githubusercontent.com/openfootball/turkey/master/2025-26/1-superlig.csv",
        "la_liga": "https://raw.githubusercontent.com/openfootball/espana/master/2025-26/1-liga.csv",
    }

    # football-data.co.uk â€“ en gÃ¼venilir tarihsel kaynak
    FOOTBALLDATA_URL = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"
    FOOTBALLDATA_LEAGUES = {
        "epl": "E0",
        "championship": "E1",
        "la_liga": "SP1",
        "bundesliga": "D1",
        "serie_a": "I1",
        "ligue_1": "F1",
        "eredivisie": "N1",
        "super_lig": "T1",
    }

    async def get_historical(self, league: str = "epl",
                              season: str = "2526") -> list[dict]:
        """football-data.co.uk'dan tarihsel sonuÃ§lar + oranlar."""
        if not HTTPX_OK:
            return []

        league_code = self.FOOTBALLDATA_LEAGUES.get(league, "E0")
        url = self.FOOTBALLDATA_URL.format(season=season, league=league_code)

        try:
            async with httpx.AsyncClient(timeout=15, verify=False) as client:
                try:
                    resp = await client.get(url)
                    if resp.status_code != 200:
                        logger.warning(
                            f"[CSV] HTTP {resp.status_code} ({league}/{season}) url={url}"
                        )
                        return []
                except httpx.TimeoutException:
                    logger.warning(f"[CSV] Timeout ({league}/{season}) url={url}")
                    return []
                except Exception as e:
                    logger.warning(f"[CSV] Bağlantı hatası: {e}")
                    return []

            text = (resp.text or "").strip()
            if not text:
                logger.warning(f"[CSV] BoÅŸ yanÄ±t ({league}/{season}) url={url}")
                return []
            # football-data zaman zaman HTML koruma sayfasÄ± dÃ¶ndÃ¼rebiliyor.
            # CSV parse denemesinden Ã¶nce kaba doÄŸrulama ile akÄ±ÅŸÄ± koru.
            first_chunk = text[:256].lower()
            if first_chunk.startswith("<!doctype") or first_chunk.startswith("<html"):
                logger.warning(
                    f"[CSV] CSV yerine HTML dÃ¶ndÃ¼ ({league}/{season}) url={url}"
                )
                return []

            if not POLARS_OK:
                return []

            df = pl.read_csv(io.StringIO(text), ignore_errors=True)

            results = []
            for row in df.iter_rows(named=True):
                entry = {
                    "source": "football_data",
                    "match_id": f"fd_{hash(str(row)) % 99999:05d}",
                    "date": row.get("Date", ""),
                    "home_team": row.get("HomeTeam", ""),
                    "away_team": row.get("AwayTeam", ""),
                    "home_goals": int(row.get("FTHG", 0) or 0),
                    "away_goals": int(row.get("FTAG", 0) or 0),
                    "result": row.get("FTR", ""),  # H/D/A
                    "league": league,
                    "season": season,
                }

                # Oranlar (varsa)
                for odds_col, key in [
                    ("B365H", "home_odds"), ("B365D", "draw_odds"), ("B365A", "away_odds"),
                    ("PSH", "pinnacle_home"), ("PSD", "pinnacle_draw"), ("PSA", "pinnacle_away"),
                    ("AvgH", "avg_home"), ("AvgD", "avg_draw"), ("AvgA", "avg_away"),
                    ("MaxH", "max_home"), ("MaxD", "max_draw"), ("MaxA", "max_away"),
                ]:
                    val = row.get(odds_col)
                    if val:
                        try:
                            entry[key] = float(val)
                        except (ValueError, TypeError):
                            pass

                # Ä°statistikler
                for stat_col, key in [
                    ("HS", "home_shots"), ("AS", "away_shots"),
                    ("HST", "home_shots_target"), ("AST", "away_shots_target"),
                    ("HC", "home_corners"), ("AC", "away_corners"),
                    ("HF", "home_fouls"), ("AF", "away_fouls"),
                    ("HY", "home_yellows"), ("AY", "away_yellows"),
                    ("HR", "home_reds"), ("AR", "away_reds"),
                ]:
                    val = row.get(stat_col)
                    if val:
                        try:
                            entry[key] = int(val)
                        except (ValueError, TypeError):
                            pass

                results.append(entry)

            logger.info(f"[CSV] {league}/{season}: {len(results)} maÃ§ + oranlar yÃ¼klendi.")
            return results
        except Exception as e:
            logger.warning(
                f"[CSV] Hata ({league}/{season}) url={url}: {type(e).__name__}: {e!r}"
            )
            return []


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  4. SOFASCORE HIDDEN API â€“ Ã‡OK SPORLU
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
class SofascoreHiddenAPI:
    """Sofascore'un iÃ§ API'sÄ±nÄ± doÄŸrudan kullan (JSON).

    Desteklenen sporlar: futbol, basketbol, tenis, voleybol, hentbol, hokey.
    TÃ¼m ligler, kupalar ve mÃ¼sabaka tÃ¼rleri dahil.
    TÃ¼rkiye'deki legal bahis sitelerinden oynanan tÃ¼m etkinlikler.
    """

    BASE_URLS = [
        "https://api.sofascore.com/api/v1",
        "https://www.sofascore.com/api/v1",
    ]

    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7",
        "Accept-Encoding": "gzip, deflate, br",
        "Origin": "https://www.sofascore.com",
        "Referer": "https://www.sofascore.com/",
        "Cache-Control": "no-cache",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-site",
        "Sec-Ch-Ua": '"Chromium";v="131", "Not_A Brand";v="24"',
        "Sec-Ch-Ua-Mobile": "?0",
        "Sec-Ch-Ua-Platform": '"Windows"',
    }
    LIGHT_HEADERS = {
        "User-Agent": HEADERS["User-Agent"],
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7",
        "Referer": "https://www.sofascore.com/",
    }

    # Sofascore sport sluglarÄ±
    SPORTS = {
        "football": "football",
        "basketball": "basketball",
        "tennis": "tennis",
        "volleyball": "volleyball",
        "handball": "handball",
        "ice-hockey": "ice-hockey",
    }

    def __init__(self):
        self._status_log_cache: dict[tuple[int, str], float] = {}
        self._status_log_ttl = 300.0

    def _log_status_once(self, status_code: int, url: str):
        key = (status_code, url)
        now = time.time()
        last = self._status_log_cache.get(key, 0.0)
        if (now - last) >= self._status_log_ttl:
            self._status_log_cache[key] = now
            logger.debug(f"[Sofascore] {status_code} {url}")

    async def _get_json(self, path: str, *, timeout: int = 12) -> dict:
        """Sofascore endpoint'lerini Ã§oklu host/header fallback ile dene."""
        if not HTTPX_OK:
            return {}
        for base in self.BASE_URLS:
            url = f"{base}{path}"
            for headers in (self.HEADERS, self.LIGHT_HEADERS):
                try:
                    async with httpx.AsyncClient(timeout=timeout, headers=headers) as client:
                        resp = await client.get(url)
                    if resp.status_code == 200:
                        return resp.json()
                    if resp.status_code in (401, 403, 429):
                        self._log_status_once(resp.status_code, url)
                        continue
                except Exception as e:
                    logger.debug(f"[Sofascore] GET hata {url}: {type(e).__name__}: {e}")
                    continue
        return {}

    # TÃ¼rkiye'deki legal bahis sitelerinden oynanan ligler/kupalar
    # (Ä°ddaa, Nesine, Misli, Bilyoner kapsamÄ±)
    PRIORITY_TOURNAMENTS = {
        "football": [
            # TÃ¼rkiye
            "SÃ¼per Lig", "1. Lig", "TFF 2. Lig", "TÃ¼rkiye KupasÄ±", "SÃ¼per Kupa",
            # Avrupa KupalarÄ±
            "UEFA Champions League", "UEFA Europa League", "UEFA Conference League",
            "UEFA Super Cup",
            # TOP-5 Avrupa
            "Premier League", "LaLiga", "Serie A", "Bundesliga", "Ligue 1",
            "FA Cup", "EFL Cup", "Copa del Rey", "DFB Pokal", "Coppa Italia",
            "Coupe de France",
            # DiÄŸer popÃ¼ler ligler
            "Eredivisie", "Primeira Liga", "Super League Greece",
            "Scottish Premiership", "Belgian Pro League",
            "Austrian Bundesliga", "Swiss Super League",
            "Russian Premier League", "Ukrainian Premier League",
            "Championship", "Serie B", "LaLiga2", "2. Bundesliga",
            # GÃ¼ney Amerika
            "Copa Libertadores", "Copa Sudamericana",
            "BrasileirÃ£o SÃ©rie A", "Liga Profesional Argentina",
            # UluslararasÄ±
            "World Cup", "European Championship", "Nations League",
            "World Cup Qualification", "Euro Qualification",
            "Copa America", "Africa Cup of Nations",
            "AFCON Qualification", "Asian Cup",
        ],
        "basketball": [
            # TÃ¼rkiye
            "BSL", "Basketbol SÃ¼per Ligi", "TBL", "TÃ¼rkiye KupasÄ±",
            # NBA & ABD
            "NBA",
            # Avrupa
            "EuroLeague", "Euroleague", "EuroCup", "Champions League",
            "FIBA Champions League",
            # DiÄŸer ligler
            "Liga ACB", "LNB Pro A", "Lega Basket Serie A",
            "Basketball Bundesliga", "Greek Basket League",
            "ABA League", "VTB United League",
            # UluslararasÄ±
            "FIBA World Cup", "EuroBasket", "Olympic Games",
        ],
        "tennis": [
            "ATP", "WTA", "Grand Slam", "Australian Open",
            "Roland Garros", "Wimbledon", "US Open",
            "ATP 1000", "ATP 500", "ATP 250",
        ],
        "volleyball": [
            "Efeler Ligi", "Sultanlar Ligi",
            "Serie A1", "Bundesliga", "SuperLega",
            "Champions League", "CEV Cup",
        ],
        "handball": [
            "Handball SÃ¼per Lig",
            "Bundesliga", "LNH Division 1", "Liga ASOBAL",
            "EHF Champions League",
        ],
        "ice-hockey": [
            "NHL", "KHL", "SHL", "DEL", "Liiga",
            "Czech Extraliga", "Swiss National League",
        ],
    }

    async def get_live_matches(self, sport: str = "football") -> list[dict]:
        """Belirli sporun canlÄ± maÃ§larÄ±nÄ± Ã§ek."""
        if not HTTPX_OK:
            return []
        sport_slug = self.SPORTS.get(sport, sport)
        data = await self._get_json(f"/sport/{sport_slug}/events/live")
        if not data:
            return []
        return self._parse_events(data.get("events", []), sport=sport)

    async def get_all_live(self) -> list[dict]:
        """TÃœM sporlarÄ±n canlÄ± maÃ§larÄ±nÄ± paralel Ã§ek."""
        if not HTTPX_OK:
            return []
        tasks = [self.get_live_matches(s) for s in self.SPORTS]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        all_events = []
        for r in results:
            if isinstance(r, list):
                all_events.extend(r)
        return all_events

    async def get_scheduled_events(self, date: str = "",
                                    sport: str = "football") -> list[dict]:
        """Belirli gÃ¼ndeki maÃ§larÄ± Ã§ek (YYYY-MM-DD)."""
        if not date:
            date = datetime.now().strftime("%Y-%m-%d")
        if not HTTPX_OK:
            return []
        sport_slug = self.SPORTS.get(sport, sport)
        data = await self._get_json(f"/sport/{sport_slug}/scheduled-events/{date}")
        if not data:
            return []
        return self._parse_events(data.get("events", []), sport=sport)

    async def get_all_scheduled(self, date: str = "") -> list[dict]:
        """TÃœM sporlarÄ±n programlÄ± maÃ§larÄ±nÄ± paralel Ã§ek."""
        if not HTTPX_OK:
            return []
        tasks = [self.get_scheduled_events(date, s) for s in self.SPORTS]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        all_events = []
        for r in results:
            if isinstance(r, list):
                all_events.extend(r)
        logger.info(f"[Sofascore] TÃ¼m sporlar: {len(all_events)} etkinlik Ã§ekildi.")
        return all_events

    async def get_match_stats(self, event_id: int) -> dict:
        """MaÃ§ istatistikleri (possession, shots, xG)."""
        if not HTTPX_OK:
            return {}
        return await self._get_json(f"/event/{event_id}/statistics")

    async def get_match_odds(self, event_id: int) -> dict:
        """MaÃ§ oranlarÄ±."""
        if not HTTPX_OK:
            return {}
        return await self._get_json(f"/event/{event_id}/odds/1/all")

    async def get_h2h(self, event_id: int) -> list[dict]:
        """Head-to-head (karÅŸÄ±lÄ±klÄ±) sonuÃ§lar."""
        if not HTTPX_OK:
            return []
        data = await self._get_json(f"/event/{event_id}/h2h")
        return self._parse_events(data.get("events", [])) if data else []

    def _is_priority_event(self, tournament_name: str, sport: str) -> bool:
        """Bu turnuva Ä°ddaa/Nesine/Misli kapsamÄ±nda mÄ±?"""
        priority = self.PRIORITY_TOURNAMENTS.get(sport, [])
        if not priority:
            return True
        tn = tournament_name.lower()
        return any(p.lower() in tn or tn in p.lower() for p in priority)

    def _detect_competition_type(self, tournament_name: str) -> str:
        """MÃ¼sabaka tÃ¼rÃ¼nÃ¼ belirle: league, cup, friendly, qualification."""
        tn = tournament_name.lower()
        if any(w in tn for w in ("cup", "kupa", "copa", "coupe", "pokal")):
            return "cup"
        if any(w in tn for w in ("qualification", "eleme", "qualifying")):
            return "qualification"
        if any(w in tn for w in ("friendly", "hazÄ±rlÄ±k", "club friendly")):
            return "friendly"
        if any(w in tn for w in ("super cup", "sÃ¼per kupa", "supercopa")):
            return "super_cup"
        return "league"

    def _parse_events(self, events: list, sport: str = "football") -> list[dict]:
        """Sofascore event listesini standart formata Ã§evir."""
        result = []
        for ev in events:
            home = ev.get("homeTeam", {})
            away = ev.get("awayTeam", {})
            home_score = ev.get("homeScore", {})
            away_score = ev.get("awayScore", {})
            tournament = ev.get("tournament", {})
            tournament_name = tournament.get("name", "")
            country = tournament.get("category", {}).get("name", "")

            result.append({
                "source": "sofascore",
                "match_id": f"ss_{ev.get('id', '')}",
                "sofascore_id": ev.get("id"),
                "sport": sport,
                "home_team": home.get("name", ""),
                "away_team": away.get("name", ""),
                "home_goals": home_score.get("current", 0) or 0,
                "away_goals": away_score.get("current", 0) or 0,
                "league": tournament_name,
                "country": country,
                "competition_type": self._detect_competition_type(tournament_name),
                "status": ev.get("status", {}).get("description", ""),
                "start_timestamp": ev.get("startTimestamp", 0),
                "slug": ev.get("slug", ""),
                "priority": self._is_priority_event(tournament_name, sport),
            })
        return result


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  5. THE ODDS API â€“ Ã‡OK SPORLU (Ãœcretsiz tier)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
class TheOddsAPI:
    """The Odds API â€“ Ã¼cretsiz tier (500 istek/ay).

    Futbol, basketbol, tenis, hokey, voleybol dahil.
    TÃ¼rkiye'deki bahis sitelerinin kapsamÄ±ndaki tÃ¼m sporlar.
    """

    BASE = "https://api.the-odds-api.com/v4"

    # Sport keys (the-odds-api format)
    SPORT_KEYS = {
        "football": [
            "soccer_turkey_super_league",
            "soccer_epl",
            "soccer_spain_la_liga",
            "soccer_germany_bundesliga",
            "soccer_italy_serie_a",
            "soccer_france_ligue_one",
            "soccer_uefa_champs_league",
            "soccer_uefa_europa_league",
            "soccer_uefa_europa_conference_league",
            "soccer_netherlands_eredivisie",
            "soccer_portugal_primeira_liga",
            "soccer_brazil_campeonato",
            "soccer_conmebol_copa_libertadores",
        ],
        "basketball": [
            "basketball_nba",
            "basketball_euroleague",
            "basketball_turkey_bsl",
        ],
        "tennis": [
            "tennis_atp_french_open",
            "tennis_atp_wimbledon",
            "tennis_atp_us_open",
            "tennis_atp_australian_open",
        ],
        "ice-hockey": [
            "icehockey_nhl",
        ],
    }

    def __init__(self, api_key: str = ""):
        self._api_key = api_key or os.environ.get("ODDS_API_KEY", "")

    async def get_all_sports(self) -> list[dict]:
        """Mevcut tÃ¼m sporlarÄ± listele."""
        if not HTTPX_OK or not self._api_key:
            return []
        try:
            async with httpx.AsyncClient(timeout=15, verify=False) as client:
                resp = await client.get(
                    f"{self.BASE}/sports",
                    params={"apiKey": self._api_key},
                )
                if resp.status_code == 200:
                    return resp.json()
        except Exception as e:
            logger.debug(f"[OddsAPI] Sports listesi hatasÄ±: {e}")
        return []

    async def get_odds(self, sport_key: str, sport_type: str = "football") -> list[dict]:
        """Belirli spor iÃ§in oranlarÄ± Ã§ek."""
        if not HTTPX_OK or not self._api_key:
            return []
        try:
            async with httpx.AsyncClient(timeout=15, verify=False) as client:
                resp = await client.get(
                    f"{self.BASE}/sports/{sport_key}/odds",
                    params={
                        "apiKey": self._api_key,
                        "regions": "eu",
                        "markets": "h2h,totals",
                        "oddsFormat": "decimal",
                    },
                )
                if resp.status_code != 200:
                    return []
                data = resp.json()
                return self._parse_odds(data, sport_type)
        except Exception as e:
            logger.debug(f"[OddsAPI] {sport_key} hatasÄ±: {e}")
            return []

    async def get_all_odds(self) -> list[dict]:
        """TÃ¼m sporlardan oranlarÄ± Ã§ek."""
        if not self._api_key:
            return []
        all_events = []
        for sport_type, keys in self.SPORT_KEYS.items():
            for key in keys[:3]:  # Her spordan max 3 lig (rate limit)
                events = await self.get_odds(key, sport_type)
                all_events.extend(events)
        logger.info(f"[OddsAPI] Toplam {len(all_events)} etkinlik oranÄ± Ã§ekildi.")
        return all_events

    def _parse_odds(self, events: list, sport_type: str) -> list[dict]:
        result = []
        for ev in events:
            home = ev.get("home_team", "")
            away = ev.get("away_team", "")
            odds_data = {}
            for bm in ev.get("bookmakers", []):
                for market in bm.get("markets", []):
                    if market.get("key") == "h2h":
                        for outcome in market.get("outcomes", []):
                            if outcome.get("name") == home:
                                odds_data["home_odds"] = outcome.get("price")
                            elif outcome.get("name") == away:
                                odds_data["away_odds"] = outcome.get("price")
                            elif outcome.get("name") == "Draw":
                                odds_data["draw_odds"] = outcome.get("price")
                break  # Ä°lk bookmaker yeterli

            result.append({
                "source": "the_odds_api",
                "match_id": f"odds_{ev.get('id', '')}",
                "sport": sport_type,
                "home_team": home,
                "away_team": away,
                "league": ev.get("sport_title", ""),
                "kickoff": ev.get("commence_time", ""),
                "status": "upcoming",
                **odds_data,
            })
        return result


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  MERKEZ: TÃœM KAYNAKLARI BÄ°RLEÅTÄ°R
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
class DataSourceAggregator:
    """TÃ¼m veri kaynaklarÄ±nÄ± tek merkezden yÃ¶neten orkestratÃ¶r.

    KullanÄ±m:
        agg = DataSourceAggregator(db=db)
        await agg.fetch_all(league="super_lig", season="2526")
    """

    def __init__(self, db=None):
        self._db = db
        self.understat = UnderstatSource()
        self.fbref = FBrefSource()
        self.csv = FootballCSVSource()
        self.sofascore = SofascoreHiddenAPI()
        self.odds_api = TheOddsAPI()
        self.mock = MockGenerator()
        self.layered = LayeredFreeProviderSystem(
            sofascore_client=self.sofascore,
            odds_client=self.odds_api,
            openfootball_urls=self.csv.LEAGUE_URLS,
        )
        logger.debug("DataSourceAggregator baÅŸlatÄ±ldÄ± (Ã§ok sporlu).")

    async def fetch_all(self, league: str = "super_lig",
                         season: str = "2526") -> dict:
        """TÃ¼m kaynaklardan paralel veri Ã§ek â€“ Ã§ok sporlu."""
        results = {}

        tasks = {
            "layered_events": self.layered.fetch_events(),
            "layered_odds": self.layered.fetch_odds(),
            "csv": self.csv.get_historical(league, season),
            "sofascore_all_scheduled": self.sofascore.get_all_scheduled(),
            "sofascore_all_live": self.sofascore.get_all_live(),
        }

        # understat sadece desteklenen liglerde
        if league in UnderstatSource.SUPPORTED_LEAGUES:
            year = f"20{season[:2]}" if len(season) == 4 else season
            tasks["understat"] = self.understat.get_league_matches(league, year)

        gathered = await asyncio.gather(
            *[asyncio.create_task(coro) for coro in tasks.values()],
            return_exceptions=True,
        )

        for name, data in zip(tasks.keys(), gathered):
            if isinstance(data, tuple) and len(data) == 2:
                rows, stats = data
                rows = rows if isinstance(rows, list) else []
                results[name] = rows
                logger.info(f"[Agg] {name}: {len(rows)} kayÄ±t | stats={stats}")
            elif isinstance(data, list):
                results[name] = data
                logger.info(f"[Agg] {name}: {len(data)} kayÄ±t")
            elif isinstance(data, Exception):
                logger.warning(f"[Agg] {name} hatasÄ±: {data}")
                results[name] = []
            else:
                results[name] = []

        # FBref senkron â€“ arka planda Ã§alÄ±ÅŸtÄ±r
        try:
            fbref_data = self.fbref.get_match_stats(league, season)
            results["fbref"] = fbref_data
            logger.info(f"[Agg] fbref: {len(fbref_data)} kayÄ±t")
        except Exception as e:
            logger.debug(f"[Agg] FBref hatasÄ±: {e}")
            results["fbref"] = []

        # DB'ye toplu kaydet
        total = 0
        if self._db:
            sport_counts: dict[str, int] = {}
            for source_data in results.values():
                for item in source_data:
                    if isinstance(item, dict) and item.get("home_team"):
                        try:
                            self._db.upsert_match(item)
                            total += 1
                            s = item.get("sport", "football")
                            sport_counts[s] = sport_counts.get(s, 0) + 1
                        except Exception:
                            pass

            # Eğer hiç veri gelmediyse MOCK verisi üret (Acil Durum Modu)
            if total == 0:
                logger.warning("[Agg] Kaynaklardan veri alınamadı -> MockGenerator devreye giriyor.")
                mock_data = self.mock.generate_live_matches(n=12)
                results["mock"] = mock_data
                for item in mock_data:
                    try:
                        self._db.upsert_match(item)
                        total += 1
                    except Exception:
                        pass
                logger.warning(f"[Agg] {total} adet MOCK veri DB'ye basıldı.")

            sport_str = ", ".join(f"{k}={v}" for k, v in sport_counts.items())
            logger.success(
                f"[Agg] Toplam {total} etkinlik DB'ye kaydedildi ({sport_str})"
            )

        return results

    async def fetch_today(self) -> list[dict]:
        """BugÃ¼nÃ¼n TÃœM sporlardan maÃ§larÄ±nÄ± Ã§ek (hÄ±zlÄ±)."""
        combined, stats = await self.layered.fetch_events()
        # Duplicates kaldÄ±r
        seen = set()
        unique = []
        for ev in combined:
            mid = ev.get("match_id", "")
            if mid and mid not in seen:
                seen.add(mid)
                unique.append(ev)
        logger.info(
            f"[Agg] BugÃ¼n tÃ¼m sporlar: {len(unique)} etkinlik "
            f"(futbol + basketbol + diÄŸer) | providers={stats}"
        )
        return unique


