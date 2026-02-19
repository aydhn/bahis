from __future__ import annotations

import asyncio
import io
import os
from datetime import datetime
from typing import Any

from loguru import logger

try:
    import httpx
    HTTPX_OK = True
except ImportError:
    HTTPX_OK = False

try:
    import polars as pl
    POLARS_OK = True
except ImportError:
    POLARS_OK = False


class APISportsProvider:
    BASE = "https://v3.football.api-sports.io"

    def __init__(self, api_key: str = ""):
        self._api_key = api_key or os.environ.get("APISPORTS_KEY", "") or os.environ.get("API_FOOTBALL_KEY", "")

    async def fetch_events(self, date: str) -> list[dict]:
        if not HTTPX_OK or not self._api_key:
            return []
        headers = {"x-apisports-key": self._api_key, "Accept": "application/json"}
        try:
            async with httpx.AsyncClient(timeout=15, headers=headers) as client:
                resp = await client.get(f"{self.BASE}/fixtures", params={"date": date})
            if resp.status_code != 200:
                return []
            rows = resp.json().get("response", [])
            out = []
            for row in rows:
                fixture = row.get("fixture", {})
                teams = row.get("teams", {})
                league = row.get("league", {})
                home = teams.get("home", {}).get("name", "")
                away = teams.get("away", {}).get("name", "")
                if not home or not away:
                    continue
                out.append(
                    {
                        "source": "api_sports",
                        "match_id": f"as_{fixture.get('id', '')}",
                        "sport": "football",
                        "league": league.get("name", ""),
                        "country": league.get("country", ""),
                        "competition_type": "league",
                        "home_team": home,
                        "away_team": away,
                        "status": fixture.get("status", {}).get("short", "upcoming"),
                        "kickoff": fixture.get("date", ""),
                        "start_timestamp": fixture.get("timestamp", 0) or 0,
                    }
                )
            return out
        except Exception as e:
            logger.debug(f"[APISports] {type(e).__name__}: {e}")
            return []


class TheSportsDBProvider:
    BASE = "https://www.thesportsdb.com/api/v1/json"
    SPORTS = [
        ("Soccer", "football"),
        ("Basketball", "basketball"),
        ("Tennis", "tennis"),
        ("Ice Hockey", "ice-hockey"),
        ("Volleyball", "volleyball"),
        ("Handball", "handball"),
    ]

    def __init__(self, api_key: str = ""):
        self._api_key = api_key or os.environ.get("THESPORTSDB_KEY", "3")

    async def fetch_events(self, date: str) -> list[dict]:
        if not HTTPX_OK:
            return []
        out: list[dict] = []
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                for sport_name, sport_slug in self.SPORTS:
                    try:
                        resp = await client.get(
                            f"{self.BASE}/{self._api_key}/eventsday.php",
                            params={"d": date, "s": sport_name},
                        )
                        if resp.status_code != 200:
                            continue
                        rows = (resp.json() or {}).get("events") or []
                        for row in rows:
                            home = row.get("strHomeTeam", "")
                            away = row.get("strAwayTeam", "")
                            if not home or not away:
                                continue
                            out.append(
                                {
                                    "source": "thesportsdb",
                                    "match_id": f"tsdb_{row.get('idEvent', '')}",
                                    "sport": sport_slug,
                                    "league": row.get("strLeague", ""),
                                    "country": row.get("strCountry", ""),
                                    "competition_type": "league",
                                    "home_team": home,
                                    "away_team": away,
                                    "status": "upcoming",
                                    "kickoff": f"{row.get('dateEvent', '')}T{row.get('strTime', '00:00:00')}",
                                }
                            )
                    except Exception as inner_e:
                        logger.debug(f"[TheSportsDB] {sport_name}: {type(inner_e).__name__}: {inner_e}")
        except Exception as e:
            logger.debug(f"[TheSportsDB] {type(e).__name__}: {e}")
        return out


class OpenLigaDBProvider:
    BASE = "https://api.openligadb.de"
    LEAGUES = ["bl1", "bl2", "bl3", "laliga", "seriea", "ligue1", "eredivisie", "premierleague"]

    async def fetch_events(self, date: str) -> list[dict]:
        if not HTTPX_OK:
            return []
        out = []
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                for code in self.LEAGUES:
                    try:
                        resp = await client.get(f"{self.BASE}/getmatchdata/{code}")
                        if resp.status_code != 200:
                            continue
                        rows = resp.json() if isinstance(resp.json(), list) else []
                        for row in rows:
                            home = (row.get("Team1") or {}).get("TeamName", "")
                            away = (row.get("Team2") or {}).get("TeamName", "")
                            if not home or not away:
                                continue
                            out.append(
                                {
                                    "source": "openligadb",
                                    "match_id": f"oldb_{row.get('MatchID', '')}",
                                    "sport": "football",
                                    "league": row.get("LeagueName", code),
                                    "country": "",
                                    "competition_type": "league",
                                    "home_team": home,
                                    "away_team": away,
                                    "status": "upcoming",
                                    "kickoff": row.get("MatchDateTimeUTC", ""),
                                }
                            )
                    except Exception as inner_e:
                        logger.debug(f"[OpenLigaDB] {code}: {type(inner_e).__name__}: {inner_e}")
        except Exception as e:
            logger.debug(f"[OpenLigaDB] {type(e).__name__}: {e}")
        return out


class OpenFootballProvider:
    def __init__(self, league_urls: dict[str, str]):
        self._league_urls = league_urls

    async def fetch_events(self, date: str) -> list[dict]:
        if not HTTPX_OK or not POLARS_OK:
            return []
        out: list[dict] = []
        try:
            async with httpx.AsyncClient(timeout=20, verify=False) as client:
                for league, url in self._league_urls.items():
                    try:
                        resp = await client.get(url)
                        if resp.status_code != 200:
                            continue
                        text = (resp.text or "").strip()
                        if not text:
                            continue
                        df = pl.read_csv(io.StringIO(text), ignore_errors=True)
                        for row in df.iter_rows(named=True):
                            home = row.get("HomeTeam") or row.get("Team 1") or row.get("home_team") or ""
                            away = row.get("AwayTeam") or row.get("Team 2") or row.get("away_team") or ""
                            dt = row.get("Date") or row.get("date") or ""
                            if not home or not away:
                                continue
                            out.append(
                                {
                                    "source": "openfootball",
                                    "match_id": f"of_{league}_{hash(f'{home}_{away}_{dt}') % 999999}",
                                    "sport": "football",
                                    "league": league,
                                    "country": "",
                                    "competition_type": "league",
                                    "home_team": str(home),
                                    "away_team": str(away),
                                    "status": "upcoming",
                                    "kickoff": str(dt),
                                }
                            )
                    except Exception as inner_e:
                        logger.debug(f"[OpenFootball] {league}: {type(inner_e).__name__}: {inner_e}")
        except Exception as e:
            logger.debug(f"[OpenFootball] {type(e).__name__}: {e}")
        return out


class FootballDataOrgProvider:
    BASE = "https://api.football-data.org/v4"
    COMPETITIONS = ["PL", "BL1", "SA", "PD", "FL1", "CL"]

    def __init__(self, api_key: str = ""):
        self._api_key = api_key or os.environ.get("FOOTBALL_DATA_API_KEY", "") or os.environ.get("FOOTBALL_DATA_KEY", "")

    async def fetch_events(self, date: str) -> list[dict]:
        if not HTTPX_OK or not self._api_key:
            return []
        out = []
        headers = {"X-Auth-Token": self._api_key}
        try:
            async with httpx.AsyncClient(timeout=15, headers=headers) as client:
                for comp in self.COMPETITIONS:
                    try:
                        resp = await client.get(
                            f"{self.BASE}/competitions/{comp}/matches",
                            params={"status": "SCHEDULED"},
                        )
                        if resp.status_code != 200:
                            continue
                        rows = (resp.json() or {}).get("matches", [])
                        for row in rows:
                            home = row.get("homeTeam", {}).get("name", "")
                            away = row.get("awayTeam", {}).get("name", "")
                            if not home or not away:
                                continue
                            out.append(
                                {
                                    "source": "football_data_org",
                                    "match_id": f"fdo_{row.get('id', '')}",
                                    "sport": "football",
                                    "league": row.get("competition", {}).get("name", comp),
                                    "country": row.get("area", {}).get("name", ""),
                                    "competition_type": "league",
                                    "home_team": home,
                                    "away_team": away,
                                    "status": row.get("status", "SCHEDULED"),
                                    "kickoff": row.get("utcDate", ""),
                                }
                            )
                    except Exception as inner_e:
                        logger.debug(f"[FD.org] {comp}: {type(inner_e).__name__}: {inner_e}")
        except Exception as e:
            logger.debug(f"[FD.org] {type(e).__name__}: {e}")
        return out


class LayeredFreeProviderSystem:
    def __init__(self, *, sofascore_client: Any, odds_client: Any, openfootball_urls: dict[str, str]):
        self._sofascore = sofascore_client
        self._odds_api = odds_client
        self._providers = [
            APISportsProvider(),
            TheSportsDBProvider(),
            OpenLigaDBProvider(),
            OpenFootballProvider(openfootball_urls),
            FootballDataOrgProvider(),
        ]

    @staticmethod
    def _dedup(events: list[dict]) -> list[dict]:
        seen = set()
        out = []
        for ev in events:
            mid = ev.get("match_id", "")
            # Some free APIs return blank/placeholder ids (e.g., "tsdb_").
            if not mid or mid.endswith("_") or len(mid) < 6:
                mid = f"{ev.get('sport','')}_{ev.get('home_team','')}_{ev.get('away_team','')}_{str(ev.get('kickoff',''))[:10]}"
            if mid in seen:
                continue
            seen.add(mid)
            out.append(ev)
        return out

    async def fetch_events(self, date: str = "") -> tuple[list[dict], dict[str, int]]:
        if not date:
            date = datetime.now().strftime("%Y-%m-%d")
        tasks = [p.fetch_events(date) for p in self._providers]
        tasks.extend([self._sofascore.get_all_scheduled(date), self._sofascore.get_all_live()])
        names = [p.__class__.__name__ for p in self._providers] + ["SofascoreScheduled", "SofascoreLive"]
        gathered = await asyncio.gather(*tasks, return_exceptions=True)
        merged: list[dict] = []
        stats: dict[str, int] = {}
        for name, data in zip(names, gathered):
            if isinstance(data, list):
                stats[name] = len(data)
                merged.extend(data)
            else:
                stats[name] = 0
                logger.debug(f"[LayeredProviders] {name} error: {data}")
        merged = [e for e in merged if isinstance(e, dict) and e.get("home_team") and e.get("away_team")]
        return self._dedup(merged), stats

    async def fetch_odds(self) -> tuple[list[dict], dict[str, int]]:
        odds = await self._odds_api.get_all_odds()
        return odds, {"TheOddsAPI": len(odds)}
