"""
lineup_monitor.py – Olay Güdümlü Kadro Takibi (Event-Driven Lineup Monitor).

Maç sonucunu en çok etkileyen faktör sahaya kimin çıkacağıdır.
Oranlar kadrolar açıklandığında sert hareket eder.

İşleyiş:
  1. Maç başlamadan 1 saat önce kaynakları (Mackolik/Sofascore) 2 dk'da bir tarar
  2. Starting XI açıklandığı an veritabanına yazar
  3. bahis.py'yi tetikleyerek analizi kadro bilgisiyle tekrar hesaplatır
  4. Yıldız oyuncu eksikse "Kadro Farkı" raporu oluşturur
"""
from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from loguru import logger

try:
    import httpx
    HTTPX_OK = True
except ImportError:
    HTTPX_OK = False

try:
    from bs4 import BeautifulSoup
    BS4_OK = True
except ImportError:
    BS4_OK = False


@dataclass
class PlayerInfo:
    """Kadrodaki bir oyuncu."""
    name: str
    position: str = ""          # GK, DEF, MID, FWD
    number: int = 0
    is_captain: bool = False
    is_star: bool = False       # Yıldız oyuncu mu? (piyasa değeri / rating)
    rating: float = 0.0         # Sofascore / WhoScored rating
    market_value: float = 0.0   # Milyon EUR


@dataclass
class LineupEvent:
    """Kadro açıklanma olayı."""
    match_id: str
    team: str
    side: str                   # home / away
    players: list[PlayerInfo] = field(default_factory=list)
    formation: str = ""         # 4-3-3, 4-4-2, etc.
    source: str = ""            # mackolik / sofascore
    timestamp: str = ""
    raw_hash: str = ""          # Değişiklik tespiti için


@dataclass
class LineupDiff:
    """İki kadro arasındaki fark."""
    match_id: str
    team: str
    missing_stars: list[str] = field(default_factory=list)
    surprise_starters: list[str] = field(default_factory=list)
    formation_changed: bool = False
    old_formation: str = ""
    new_formation: str = ""
    impact_score: float = 0.0   # -1.0 (çok kötü) ... +1.0 (çok iyi)


class LineupMonitor:
    """Olay güdümlü kadro tarayıcı.

    Kullanım:
        monitor = LineupMonitor(db=db, notifier=notifier)
        await monitor.watch(match_ids, shutdown_event)
    """

    POLL_INTERVAL = 120     # 2 dakika
    STAR_THRESHOLD = 7.0    # Rating >= 7.0 → yıldız oyuncu

    def __init__(self, db=None, notifier=None, cb_registry=None,
                 on_lineup_detected=None):
        self._db = db
        self._notifier = notifier
        self._cb_registry = cb_registry
        self._on_lineup = on_lineup_detected  # async callback(LineupEvent)
        self._known_lineups: dict[str, str] = {}  # match_team → hash
        self._star_players: dict[str, list[str]] = {}  # team → [star names]
        self._last_events: list[LineupEvent] = []
        logger.debug("LineupMonitor başlatıldı.")

    # ═══════════════════════════════════════════
    #  ANA DÖNGÜ
    # ═══════════════════════════════════════════
    async def watch(self, shutdown: asyncio.Event):
        """Yaklaşan maçları izle, kadro açıklanınca tetikle."""
        logger.info("[Lineup] Kadro izleme başladı.")
        while not shutdown.is_set():
            try:
                matches = self._get_upcoming_matches()
                for match in matches:
                    match_id = match.get("match_id", "")
                    home = match.get("home_team", "")
                    away = match.get("away_team", "")
                    kickoff = match.get("kickoff", "")

                    if not self._should_monitor(kickoff):
                        continue

                    for team, side in [(home, "home"), (away, "away")]:
                        event = await self._fetch_lineup(match_id, team, side)
                        if event and self._is_new_or_changed(event):
                            await self._process_lineup(event, match)

            except Exception as e:
                logger.error(f"[Lineup] Tarama hatası: {e}")

            await asyncio.sleep(self.POLL_INTERVAL)

    def _should_monitor(self, kickoff: str) -> bool:
        """Maç 2 saat içinde mi? (kadro genelde 1 saat önce açıklanır)."""
        try:
            if isinstance(kickoff, str):
                kick_dt = datetime.fromisoformat(kickoff)
            else:
                kick_dt = kickoff
            now = datetime.now()
            return timedelta(0) < (kick_dt - now) < timedelta(hours=2)
        except (ValueError, TypeError):
            return False

    def _get_upcoming_matches(self) -> list[dict]:
        """DB'den yaklaşan maçları al."""
        if self._db and hasattr(self._db, "get_upcoming_matches"):
            try:
                df = self._db.get_upcoming_matches()
                return df.to_dicts() if hasattr(df, "to_dicts") else []
            except Exception:
                pass
        return []

    # ═══════════════════════════════════════════
    #  KADRO ÇEKME
    # ═══════════════════════════════════════════
    async def _fetch_lineup(self, match_id: str, team: str,
                            side: str) -> LineupEvent | None:
        """Kaynaktan kadro bilgisini çek."""
        event = await self._fetch_sofascore(match_id, team, side)
        if not event:
            event = await self._fetch_mackolik(match_id, team, side)
        return event

    async def _fetch_sofascore(self, match_id: str, team: str,
                                side: str) -> LineupEvent | None:
        """Sofascore'dan kadro çek (API tabanlı)."""
        if not HTTPX_OK:
            return None
        try:
            from fake_useragent import UserAgent
            ua = UserAgent()
            headers = {"User-Agent": ua.random}
        except ImportError:
            headers = {"User-Agent": "Mozilla/5.0"}

        try:
            async with httpx.AsyncClient(timeout=15, headers=headers) as client:
                url = f"https://api.sofascore.com/api/v1/event/{match_id}/lineups"
                resp = await client.get(url)
                if resp.status_code != 200:
                    return None

                data = resp.json()
                lineup_data = data.get("home" if side == "home" else "away", {})
                players_raw = lineup_data.get("players", [])
                formation = lineup_data.get("formation", "")

                players = []
                for p in players_raw:
                    player = p.get("player", {})
                    players.append(PlayerInfo(
                        name=player.get("name", ""),
                        position=player.get("position", ""),
                        number=player.get("jerseyNumber", 0),
                        is_captain=p.get("captain", False),
                        rating=p.get("statistics", {}).get("rating", 0),
                    ))

                raw_str = f"{formation}|{'|'.join(p.name for p in players)}"
                return LineupEvent(
                    match_id=match_id,
                    team=team,
                    side=side,
                    players=players,
                    formation=formation,
                    source="sofascore",
                    timestamp=datetime.utcnow().isoformat(),
                    raw_hash=hashlib.md5(raw_str.encode()).hexdigest(),
                )
        except Exception as e:
            logger.debug(f"[Lineup] Sofascore hatası: {e}")
            return None

    async def _fetch_mackolik(self, match_id: str, team: str,
                               side: str) -> LineupEvent | None:
        """Mackolik'ten kadro çek (HTML parsing)."""
        if not (HTTPX_OK and BS4_OK):
            return None
        try:
            from fake_useragent import UserAgent
            ua = UserAgent()
            headers = {"User-Agent": ua.random}
        except ImportError:
            headers = {"User-Agent": "Mozilla/5.0"}

        try:
            async with httpx.AsyncClient(timeout=15, headers=headers) as client:
                url = f"https://www.mackolik.com/mac/{match_id}/kadro"
                resp = await client.get(url)
                if resp.status_code != 200:
                    return None

                soup = BeautifulSoup(resp.text, "lxml")
                lineup_div = soup.select_one(
                    f".lineup-{'home' if side == 'home' else 'away'}"
                )
                if not lineup_div:
                    return None

                players = []
                for row in lineup_div.select(".player-row"):
                    name = row.select_one(".player-name")
                    pos = row.select_one(".player-position")
                    if name:
                        players.append(PlayerInfo(
                            name=name.text.strip(),
                            position=pos.text.strip() if pos else "",
                        ))

                formation_el = lineup_div.select_one(".formation")
                formation = formation_el.text.strip() if formation_el else ""

                raw_str = f"{formation}|{'|'.join(p.name for p in players)}"
                return LineupEvent(
                    match_id=match_id,
                    team=team,
                    side=side,
                    players=players,
                    formation=formation,
                    source="mackolik",
                    timestamp=datetime.utcnow().isoformat(),
                    raw_hash=hashlib.md5(raw_str.encode()).hexdigest(),
                )
        except Exception as e:
            logger.debug(f"[Lineup] Mackolik hatası: {e}")
            return None

    # ═══════════════════════════════════════════
    #  DEĞİŞİKLİK TESPİTİ
    # ═══════════════════════════════════════════
    def _is_new_or_changed(self, event: LineupEvent) -> bool:
        """Kadro yeni mi yoksa değişti mi?"""
        key = f"{event.match_id}_{event.team}"
        old_hash = self._known_lineups.get(key)
        if old_hash == event.raw_hash:
            return False
        self._known_lineups[key] = event.raw_hash
        return True

    async def _process_lineup(self, event: LineupEvent, match: dict):
        """Kadro algılandı – işle, kaydet, bildir."""
        logger.info(
            f"[Lineup] 🔔 KADRO TESPİT: {event.team} ({event.side}) "
            f"– {event.formation} – {len(event.players)} oyuncu"
        )

        # Yıldız oyuncu kontrolü
        diff = self._detect_star_absence(event)

        # Veritabanına kaydet
        if self._db and hasattr(self._db, "save_lineup"):
            try:
                self._db.save_lineup(event)
            except Exception:
                pass

        # Callback: bahis.py'yi tetikle → analizi yeniden hesaplat
        if self._on_lineup:
            try:
                await self._on_lineup(event)
            except Exception as e:
                logger.error(f"[Lineup] Callback hatası: {e}")

        # Telegram bildirimi
        if self._notifier:
            await self._notify_lineup(event, diff)

        self._last_events.append(event)

    def _detect_star_absence(self, event: LineupEvent) -> LineupDiff | None:
        """Yıldız oyuncunun eksikliğini tespit et."""
        known_stars = self._star_players.get(event.team, [])
        if not known_stars:
            return None

        current_names = {p.name.lower() for p in event.players}
        missing = [s for s in known_stars if s.lower() not in current_names]

        if not missing:
            return None

        impact = -0.15 * len(missing)  # Her eksik yıldız -%15 etki
        diff = LineupDiff(
            match_id=event.match_id,
            team=event.team,
            missing_stars=missing,
            impact_score=max(impact, -1.0),
        )
        logger.warning(
            f"[Lineup] ⚠️ YILDIZ EKSİK: {event.team} – "
            f"{', '.join(missing)} (etki: {diff.impact_score:+.0%})"
        )
        return diff

    def register_star_players(self, team: str, stars: list[str]):
        """Bir takımın yıldız oyuncularını kaydet."""
        self._star_players[team] = stars

    async def _notify_lineup(self, event: LineupEvent,
                              diff: LineupDiff | None):
        """Telegram'a kadro bildirimi gönder."""
        xi = "\n".join(
            f"  {'⭐' if p.is_captain else '•'} {p.name} ({p.position})"
            for p in event.players[:11]
        )

        text = (
            f"📋 <b>KADRO AÇIKLANDI</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"🏟 <b>Maç:</b> {event.match_id}\n"
            f"👥 <b>Takım:</b> {event.team} ({event.side.upper()})\n"
            f"📐 <b>Diziliş:</b> {event.formation or 'Bilinmiyor'}\n"
            f"📡 <b>Kaynak:</b> {event.source}\n\n"
            f"<b>İlk 11:</b>\n{xi}"
        )

        if diff and diff.missing_stars:
            text += (
                f"\n\n⚠️ <b>YILDIZ EKSİK!</b>\n"
                f"{''.join(f'  ❌ {s}' + chr(10) for s in diff.missing_stars)}"
                f"📉 <b>Etki Skoru:</b> {diff.impact_score:+.0%}"
            )

        await self._notifier.send(text)

    @property
    def last_events(self) -> list[LineupEvent]:
        return self._last_events[-20:]
