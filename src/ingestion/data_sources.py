"""
data_sources.py – Ücretsiz Ham ve Temiz Veri Kaynakları ("Gizli Hazineler").

Resmi API'lar (Opta, Sportradar) aylık binlerce dolar ister.
Bir Quant Developer arka kapıları bilir:

1. understat       → xG verileri (dünyadaki en iyi ücretsiz kaynak)
2. soccerdata/FBref → Detaylı maç istatistikleri (şut, pas, koşu mesafesi)
3. football-csv     → Tüm liglerin tarihsel sonuçları (CSV)
4. Hidden APIs      → Sofascore/Flashscore JSON endpoint'leri

Tüm kaynaklar → Polars DataFrame → DuckDB veritabanına akış.
"""
from __future__ import annotations

import asyncio
import io
import time
from datetime import datetime
from typing import Any

from loguru import logger

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


# ═══════════════════════════════════════════════
#  1. UNDERSTAT – xG VERİLERİ
# ═══════════════════════════════════════════════
class UnderstatSource:
    """understat.com – xG verileri (pip install understat).

    Dünyadaki en kapsamlı ücretsiz xG kaynağı.
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
        """Lig maçlarını xG verileriyle çek."""
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
                logger.info(f"[Understat] {league}/{season}: {len(result)} maç çekildi.")
                return result
        except ImportError:
            logger.warning("[Understat] understat kütüphanesi yüklü değil: pip install understat")
            return []
        except Exception as e:
            logger.error(f"[Understat] Hata: {e}")
            return []

    async def get_team_xg(self, team: str, league: str = "epl",
                           season: str = "2025") -> list[dict]:
        """Belirli bir takımın maç bazlı xG verileri."""
        try:
            import understat
            async with understat.UnderstatClient() as client:
                players = await client.get_team_stats(team, int(season))
                return players if isinstance(players, list) else []
        except Exception as e:
            logger.debug(f"[Understat] Takım xG hatası: {e}")
            return []

    def _check_cache(self, key: str):
        entry = self._cache.get(key)
        if entry and (time.time() - entry[0]) < self._cache_ttl:
            return entry[1]
        return None

    def _set_cache(self, key: str, data: Any):
        self._cache[key] = (time.time(), data)


# ═══════════════════════════════════════════════
#  2. SOCCERDATA / FBREF – DETAYLI İSTATİSTİK
# ═══════════════════════════════════════════════
class FBrefSource:
    """FBref (via soccerdata) – Detaylı maç istatistikleri.

    şut, pas, koşu mesafesi, possession, corners...
    pip install soccerdata
    """

    LEAGUE_MAP = {
        "super_lig": "TUR-Süper Lig",
        "epl": "ENG-Premier League",
        "la_liga": "ESP-La Liga",
        "bundesliga": "GER-Bundesliga",
        "serie_a": "ITA-Serie A",
    }

    def __init__(self):
        self._ws = None

    def get_match_stats(self, league: str = "epl",
                         season: str = "2526") -> list[dict]:
        """FBref'ten detaylı maç istatistikleri çek."""
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

            logger.info(f"[FBref] {league}/{season}: {len(results)} maç.")
            return results
        except ImportError:
            logger.warning("[FBref] soccerdata yüklü değil: pip install soccerdata")
            return []
        except Exception as e:
            logger.error(f"[FBref] Hata: {e}")
            return []

    def get_shooting_stats(self, league: str = "epl",
                            season: str = "2526") -> list[dict]:
        """Şut istatistikleri (xG, npxG, key passes)."""
        try:
            import soccerdata as sd
            fbref = sd.FBref(leagues=self.LEAGUE_MAP.get(league, league),
                             seasons=season)
            shooting = fbref.read_team_season_stats(stat_type="shooting")
            if shooting is None or shooting.empty:
                return []

            return shooting.reset_index().to_dict("records")
        except Exception as e:
            logger.debug(f"[FBref] Shooting stats hatası: {e}")
            return []


# ═══════════════════════════════════════════════
#  3. FOOTBALL-CSV – TARİHSEL SONUÇLAR
# ═══════════════════════════════════════════════
class FootballCSVSource:
    """football-csv (GitHub) – Tüm liglerin CSV sonuçları.

    Doğrudan URL'den okuma – API kurulumu gerekmez.
    """

    BASE_URL = "https://raw.githubusercontent.com/footballcsv/footballcsv.github.io/master"

    # Popüler lig dosya yolları (örnekler)
    LEAGUE_URLS = {
        "epl": "https://raw.githubusercontent.com/openfootball/england/master/2025-26/1-premierleague.csv",
        "super_lig": "https://raw.githubusercontent.com/openfootball/turkey/master/2025-26/1-superlig.csv",
        "la_liga": "https://raw.githubusercontent.com/openfootball/espana/master/2025-26/1-liga.csv",
    }

    # football-data.co.uk – en güvenilir tarihsel kaynak
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
        """football-data.co.uk'dan tarihsel sonuçlar + oranlar."""
        if not HTTPX_OK:
            return []

        league_code = self.FOOTBALLDATA_LEAGUES.get(league, "E0")
        url = self.FOOTBALLDATA_URL.format(season=season, league=league_code)

        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(url)
                if resp.status_code != 200:
                    logger.debug(f"[CSV] {url} → {resp.status_code}")
                    return []

            if not POLARS_OK:
                return []

            df = pl.read_csv(io.StringIO(resp.text), ignore_errors=True)

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

                # İstatistikler
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

            logger.info(f"[CSV] {league}/{season}: {len(results)} maç + oranlar yüklendi.")
            return results
        except Exception as e:
            logger.error(f"[CSV] Hata: {e}")
            return []


# ═══════════════════════════════════════════════
#  4. SOFASCORE HIDDEN API
# ═══════════════════════════════════════════════
class SofascoreHiddenAPI:
    """Sofascore'un iç API'sını doğrudan kullan (JSON).

    HTML parsing yerine saf JSON – hızlı ve güvenilir.
    """

    BASE = "https://api.sofascore.com/api/v1"

    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json",
        "Referer": "https://www.sofascore.com/",
    }

    async def get_live_matches(self) -> list[dict]:
        """Canlı maçları çek."""
        if not HTTPX_OK:
            return []
        try:
            async with httpx.AsyncClient(timeout=10, headers=self.HEADERS) as client:
                resp = await client.get(f"{self.BASE}/sport/football/events/live")
                if resp.status_code != 200:
                    return []
                data = resp.json()
                return self._parse_events(data.get("events", []))
        except Exception as e:
            logger.debug(f"[Sofascore] Live hatası: {e}")
            return []

    async def get_scheduled_events(self, date: str = "") -> list[dict]:
        """Belirli gündeki maçları çek (YYYY-MM-DD)."""
        if not date:
            date = datetime.now().strftime("%Y-%m-%d")
        if not HTTPX_OK:
            return []
        try:
            async with httpx.AsyncClient(timeout=10, headers=self.HEADERS) as client:
                resp = await client.get(f"{self.BASE}/sport/football/scheduled-events/{date}")
                if resp.status_code != 200:
                    return []
                data = resp.json()
                return self._parse_events(data.get("events", []))
        except Exception as e:
            logger.debug(f"[Sofascore] Schedule hatası: {e}")
            return []

    async def get_match_stats(self, event_id: int) -> dict:
        """Maç istatistikleri (possession, shots, xG)."""
        if not HTTPX_OK:
            return {}
        try:
            async with httpx.AsyncClient(timeout=10, headers=self.HEADERS) as client:
                resp = await client.get(f"{self.BASE}/event/{event_id}/statistics")
                if resp.status_code != 200:
                    return {}
                return resp.json()
        except Exception:
            return {}

    async def get_match_odds(self, event_id: int) -> dict:
        """Maç oranları."""
        if not HTTPX_OK:
            return {}
        try:
            async with httpx.AsyncClient(timeout=10, headers=self.HEADERS) as client:
                resp = await client.get(f"{self.BASE}/event/{event_id}/odds/1/all")
                if resp.status_code != 200:
                    return {}
                return resp.json()
        except Exception:
            return {}

    async def get_h2h(self, event_id: int) -> list[dict]:
        """Head-to-head (karşılıklı) sonuçlar."""
        if not HTTPX_OK:
            return []
        try:
            async with httpx.AsyncClient(timeout=10, headers=self.HEADERS) as client:
                resp = await client.get(f"{self.BASE}/event/{event_id}/h2h")
                if resp.status_code != 200:
                    return []
                data = resp.json()
                return self._parse_events(data.get("events", []))
        except Exception:
            return []

    def _parse_events(self, events: list) -> list[dict]:
        """Sofascore event listesini standart formata çevir."""
        result = []
        for ev in events:
            home = ev.get("homeTeam", {})
            away = ev.get("awayTeam", {})
            home_score = ev.get("homeScore", {})
            away_score = ev.get("awayScore", {})
            tournament = ev.get("tournament", {})

            result.append({
                "source": "sofascore",
                "match_id": f"ss_{ev.get('id', '')}",
                "sofascore_id": ev.get("id"),
                "home_team": home.get("name", ""),
                "away_team": away.get("name", ""),
                "home_goals": home_score.get("current", 0) or 0,
                "away_goals": away_score.get("current", 0) or 0,
                "league": tournament.get("name", ""),
                "country": tournament.get("category", {}).get("name", ""),
                "status": ev.get("status", {}).get("description", ""),
                "start_timestamp": ev.get("startTimestamp", 0),
                "slug": ev.get("slug", ""),
            })
        return result


# ═══════════════════════════════════════════════
#  MERKEZ: TÜM KAYNAKLARI BİRLEŞTİR
# ═══════════════════════════════════════════════
class DataSourceAggregator:
    """Tüm veri kaynaklarını tek merkezden yöneten orkestratör.

    Kullanım:
        agg = DataSourceAggregator(db=db)
        await agg.fetch_all(league="super_lig", season="2526")
    """

    def __init__(self, db=None):
        self._db = db
        self.understat = UnderstatSource()
        self.fbref = FBrefSource()
        self.csv = FootballCSVSource()
        self.sofascore = SofascoreHiddenAPI()
        logger.debug("DataSourceAggregator başlatıldı.")

    async def fetch_all(self, league: str = "super_lig",
                         season: str = "2526") -> dict:
        """Tüm kaynaklardan paralel veri çek."""
        results = {}

        tasks = {
            "csv": self.csv.get_historical(league, season),
            "sofascore_live": self.sofascore.get_live_matches(),
            "sofascore_today": self.sofascore.get_scheduled_events(),
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
            if isinstance(data, list):
                results[name] = data
                logger.info(f"[Agg] {name}: {len(data)} kayıt")
            elif isinstance(data, Exception):
                logger.warning(f"[Agg] {name} hatası: {data}")
                results[name] = []
            else:
                results[name] = []

        # FBref senkron – arka planda çalıştır
        try:
            fbref_data = self.fbref.get_match_stats(league, season)
            results["fbref"] = fbref_data
            logger.info(f"[Agg] fbref: {len(fbref_data)} kayıt")
        except Exception as e:
            logger.debug(f"[Agg] FBref hatası: {e}")
            results["fbref"] = []

        # DB'ye toplu kaydet
        if self._db:
            total = 0
            for source_data in results.values():
                for item in source_data:
                    if isinstance(item, dict) and item.get("home_team"):
                        try:
                            self._db.upsert_match(item)
                            total += 1
                        except Exception as e:
                            logger.debug(f"Exception caught: {e}")
            logger.success(f"[Agg] Toplam {total} maç veritabanına kaydedildi.")

        return results

    async def fetch_today(self) -> list[dict]:
        """Bugünün maçlarını çek (hızlı)."""
        live = await self.sofascore.get_live_matches()
        scheduled = await self.sofascore.get_scheduled_events()
        return live + scheduled
