"""
api_hijacker.py – Network Sniffing Agent (Ağ Koklayıcı).

HTML Scraping devri bitti. Sitelere "kullanıcı gibi" görünüp
arka plandaki saf veri akışını (WebSocket/XHR) dinliyoruz.

Yöntem:
  1. Playwright ile hedef siteyi aç
  2. Network trafiğini dinle (page.on("response"))
  3. WebSocket mesajlarını yakala (canlı skor akışı)
  4. api/v1/event içeren JSON yanıtlarını filtrele
  5. Temiz veriyi veritabanına kaydet

Avantaj: BeautifulSoup ile HTML parse etmek kırılgandır.
Doğrudan JSON kaynağına bağlanmak 10x hızlı ve stabil.
"""
from __future__ import annotations

import asyncio
import json
import re
import time
from datetime import datetime
from typing import Any

from loguru import logger


class APIHijacker:
    """Ağ trafiğini dinleyerek gizli API endpoint'lerini keşfeder.

    Kullanım:
        hijacker = APIHijacker(db=db)
        await hijacker.listen(shutdown_event)

    Yakalanan endpoint'ler otomatik olarak keşfedilir ve
    sonraki seferlerde doğrudan kullanılır (öğrenen sistem).
    """

    # Yakalanacak URL pattern'leri
    URL_PATTERNS = [
        re.compile(r"api/v\d+/event", re.IGNORECASE),
        re.compile(r"api/v\d+/sport/football", re.IGNORECASE),
        re.compile(r"odds", re.IGNORECASE),
        re.compile(r"match.*data", re.IGNORECASE),
        re.compile(r"live.*score", re.IGNORECASE),
        re.compile(r"event.*list", re.IGNORECASE),
        re.compile(r"fixture", re.IGNORECASE),
        re.compile(r"lineups?", re.IGNORECASE),
        re.compile(r"statistics", re.IGNORECASE),
        re.compile(r"standings", re.IGNORECASE),
        re.compile(r"h2h", re.IGNORECASE),
    ]

    # İzlenecek hedef siteler
    TARGET_SITES = [
        {
            "url": "https://www.sofascore.com/football",
            "name": "sofascore",
            "wait_ms": 5000,
        },
        {
            "url": "https://www.sofascore.com/basketball",
            "name": "sofascore_basketball",
            "wait_ms": 5000,
        },
        {
            "url": "https://www.sofascore.com/tennis",
            "name": "sofascore_tennis",
            "wait_ms": 4000,
        },
        {
            "url": "https://www.flashscore.com.tr/",
            "name": "flashscore",
            "wait_ms": 3000,
        },
        {
            "url": "https://www.mackolik.com/canli-sonuclar",
            "name": "mackolik",
            "wait_ms": 3000,
        },
    ]

    # Gürültü filtresi: bu pattern'leri ATLAT
    IGNORE_PATTERNS = [
        re.compile(r"google", re.IGNORECASE),
        re.compile(r"analytics", re.IGNORECASE),
        re.compile(r"facebook", re.IGNORECASE),
        re.compile(r"advert", re.IGNORECASE),
        re.compile(r"tracking", re.IGNORECASE),
        re.compile(r"\.(png|jpg|gif|svg|css|woff|ico)", re.IGNORECASE),
    ]

    def __init__(self, db=None, cb_registry=None):
        self._db = db
        self._cb_registry = cb_registry
        self._captured: list[dict] = []
        self._discovered_endpoints: dict[str, dict] = {}  # URL → meta
        self._ws_messages: list[dict] = []
        self._stats = {
            "total_requests": 0,
            "matched_requests": 0,
            "json_captured": 0,
            "ws_frames": 0,
            "matches_stored": 0,
        }
        logger.debug("APIHijacker v2 (Network Sniffer) başlatıldı.")

    # ═══════════════════════════════════════════
    #  ANA DİNLEYİCİ
    # ═══════════════════════════════════════════
    async def listen(self, shutdown: asyncio.Event):
        """Playwright ile hedef siteleri dinlemeye başla."""
        logger.info("[Hijack] Ağ dinleyici başlatılıyor…")

        try:
            from playwright.async_api import async_playwright
        except ImportError:
            logger.warning("[Hijack] Playwright yüklü değil – devre dışı.")
            return

        try:
            pw = await async_playwright().start()
            browser = await pw.chromium.launch(
                headless=True,
                args=["--disable-blink-features=AutomationControlled"],
            )
        except Exception as e:
            logger.warning(f"[Hijack] Tarayıcı başlatılamadı: {e}")
            return

        # Her site için bir sayfa aç ve dinle
        pages = []
        for target in self.TARGET_SITES:
            try:
                context = await browser.new_context(
                    user_agent=(
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0.0.0 Safari/537.36"
                    ),
                    viewport={"width": 1920, "height": 1080},
                    locale="tr-TR",
                )
                page = await context.new_page()

                # Ağ trafiği dinleyicileri
                page.on("response", lambda resp, t=target: asyncio.ensure_future(
                    self._on_response(resp, t["name"])
                ))
                page.on("websocket", lambda ws, t=target: self._on_websocket(ws, t["name"]))

                await page.goto(target["url"], timeout=30000, wait_until="networkidle")
                await asyncio.sleep(target.get("wait_ms", 3000) / 1000)

                pages.append(page)
                logger.info(f"[Hijack] ✓ Dinleniyor: {target['name']} ({target['url']})")
            except Exception as e:
                if shutdown.is_set():
                    logger.info(f"[Hijack] {target['name']} kapanışta atlandı.")
                else:
                    logger.warning(f"[Hijack] ✗ {target['name']} bağlanamadı: {e}")

        # Periyodik yenileme ile dinlemeye devam et
        refresh_cycle = 0
        while not shutdown.is_set():
            await asyncio.sleep(30)
            refresh_cycle += 1

            # Her 5 dakikada sayfaları yenile (taze veri)
            if refresh_cycle % 10 == 0:
                for page in pages:
                    try:
                        await page.reload(timeout=15000)
                    except Exception:
                        pass
                logger.debug(
                    f"[Hijack] Durum: {self._stats['json_captured']} JSON, "
                    f"{self._stats['ws_frames']} WS, "
                    f"{self._stats['matches_stored']} maç"
                )

        # Temizlik
        for page in pages:
            try:
                await page.close()
            except Exception:
                pass
        await browser.close()
        logger.info(
            f"[Hijack] Kapatıldı. Son durum: "
            f"{self._stats['json_captured']} JSON yakalandı, "
            f"{len(self._discovered_endpoints)} endpoint keşfedildi."
        )

    # ═══════════════════════════════════════════
    #  RESPONSE HANDLER
    # ═══════════════════════════════════════════
    async def _on_response(self, response, source: str):
        """Her HTTP yanıtını kontrol et."""
        self._stats["total_requests"] += 1
        url = response.url

        # Gürültü filtresi
        if any(p.search(url) for p in self.IGNORE_PATTERNS):
            return

        # Pattern eşleşmesi
        if not any(p.search(url) for p in self.URL_PATTERNS):
            return

        self._stats["matched_requests"] += 1

        try:
            content_type = response.headers.get("content-type", "")
            if "json" not in content_type and "javascript" not in content_type:
                return

            body = await response.json()
            self._stats["json_captured"] += 1

            # Endpoint keşfi
            self._discover_endpoint(url, source, response.status, body)

            # Veri işleme
            record = {
                "url": url,
                "source": source,
                "status": response.status,
                "data": body,
                "timestamp": datetime.utcnow().isoformat(),
                "size": len(str(body)),
            }
            self._captured.append(record)

            # Veriyi parse et ve DB'ye kaydet
            matches_found = self._extract_and_store(body, source, url)
            if matches_found > 0:
                logger.info(
                    f"[Hijack] 🎯 {source}: {url[:80]}… → {matches_found} maç"
                )

        except Exception:
            pass

    # ═══════════════════════════════════════════
    #  WEBSOCKET HANDLER
    # ═══════════════════════════════════════════
    def _on_websocket(self, ws, source: str):
        """WebSocket bağlantısını dinle."""
        logger.debug(f"[Hijack] WebSocket açıldı: {source} → {ws.url[:80]}")

        def on_frame(payload):
            self._stats["ws_frames"] += 1
            try:
                data = json.loads(payload)
                self._ws_messages.append({
                    "source": source,
                    "data": data,
                    "timestamp": datetime.utcnow().isoformat(),
                })

                # Canlı skor güncellemesi?
                if isinstance(data, dict):
                    self._extract_and_store(data, f"{source}_ws", "websocket")
            except (json.JSONDecodeError, TypeError):
                pass

        ws.on("framereceived", lambda f: on_frame(f))

    # ═══════════════════════════════════════════
    #  ENDPOINT KEŞFİ (Öğrenen Sistem)
    # ═══════════════════════════════════════════
    def _discover_endpoint(self, url: str, source: str,
                            status: int, body: Any):
        """Yeni API endpoint'leri keşfet ve katalogla."""
        # URL'yi template'e çevir (ID'leri * ile değiştir)
        template = re.sub(r"/\d+", "/*", url.split("?")[0])

        if template not in self._discovered_endpoints:
            # Veri yapısını analiz et
            data_type = self._classify_data(body)

            self._discovered_endpoints[template] = {
                "template": template,
                "source": source,
                "first_seen": datetime.utcnow().isoformat(),
                "hit_count": 0,
                "data_type": data_type,
                "sample_url": url,
                "status": status,
            }
            logger.info(
                f"[Hijack] 🆕 Endpoint keşfedildi: {template} "
                f"[{data_type}] ({source})"
            )

        self._discovered_endpoints[template]["hit_count"] += 1

    @staticmethod
    def _classify_data(body: Any) -> str:
        """Yakalanan verinin türünü sınıfla."""
        if not isinstance(body, dict):
            return "unknown"

        keys = set(str(k).lower() for k in body.keys())

        if "events" in keys or "matches" in keys:
            return "match_list"
        elif "odds" in keys:
            return "odds"
        elif "statistics" in keys or "stats" in keys:
            return "statistics"
        elif "lineups" in keys or "lineup" in keys:
            return "lineups"
        elif "standings" in keys or "table" in keys:
            return "standings"
        elif "h2h" in keys:
            return "head_to_head"
        elif any(k in keys for k in ("hometeam", "home", "homescore")):
            return "match_detail"
        return "other"

    # ═══════════════════════════════════════════
    #  VERİ ÇIKARMA & KAYIT
    # ═══════════════════════════════════════════
    def _extract_and_store(self, data: Any, source: str,
                            url: str) -> int:
        """JSON verisinden maç bilgilerini çıkar ve DB'ye kaydet."""
        count = 0

        if isinstance(data, dict):
            # Event listesi
            events = (
                data.get("events") or data.get("matches") or
                data.get("d") or data.get("data") or
                data.get("results") or []
            )

            if isinstance(events, list):
                for item in events[:100]:
                    if self._try_store_match(item, source):
                        count += 1

            # Tek maç detayı
            if not events and self._try_store_match(data, source):
                count += 1

        elif isinstance(data, list):
            for item in data[:100]:
                if self._try_store_match(item, source):
                    count += 1

        self._stats["matches_stored"] += count
        return count

    def _try_store_match(self, item: Any, source: str) -> bool:
        """Tek bir maç verisini parse edip DB'ye kaydet."""
        if not isinstance(item, dict):
            return False

        # Takım isimleri – farklı formatları destekle
        home = (
            item.get("homeTeam", {}).get("name", "") if isinstance(item.get("homeTeam"), dict)
            else item.get("HN") or item.get("home") or item.get("homeTeam", "")
        )
        away = (
            item.get("awayTeam", {}).get("name", "") if isinstance(item.get("awayTeam"), dict)
            else item.get("AN") or item.get("away") or item.get("awayTeam", "")
        )

        if not home or not away:
            return False

        match_id = f"hjk_{source}_{hash(f'{home}{away}') % 99999:05d}"

        match_data = {
            "match_id": match_id,
            "source": source,
            "home_team": str(home),
            "away_team": str(away),
            "league": (
                item.get("tournament", {}).get("name", "")
                if isinstance(item.get("tournament"), dict)
                else item.get("LN") or item.get("league", "Unknown")
            ),
            "kickoff": (
                item.get("startTimestamp") or
                item.get("DT") or item.get("date") or
                datetime.utcnow().isoformat()
            ),
            "status": (
                item.get("status", {}).get("description", "upcoming")
                if isinstance(item.get("status"), dict)
                else item.get("status", "upcoming")
            ),
        }

        # Skor
        home_score = item.get("homeScore", {})
        away_score = item.get("awayScore", {})
        if isinstance(home_score, dict):
            match_data["home_goals"] = home_score.get("current", 0) or 0
            match_data["away_goals"] = away_score.get("current", 0) or 0
        elif "FTHG" in item:
            match_data["home_goals"] = int(item.get("FTHG", 0) or 0)
            match_data["away_goals"] = int(item.get("FTAG", 0) or 0)

        # Oranlar
        for key_src, key_dst in [
            ("HO", "home_odds"), ("DO", "draw_odds"), ("AO", "away_odds"),
        ]:
            val = item.get(key_src)
            if val:
                try:
                    match_data[key_dst] = float(val)
                except (ValueError, TypeError):
                    pass

        if self._db:
            try:
                self._db.upsert_match(match_data)
                return True
            except Exception:
                pass
        return False

    # ═══════════════════════════════════════════
    #  API DIRECT ACCESS (Keşfedilen endpoint'leri kullan)
    # ═══════════════════════════════════════════
    async def direct_fetch(self, endpoint_template: str,
                            params: dict | None = None) -> Any:
        """Daha önce keşfedilen endpoint'e doğrudan istek at."""
        meta = self._discovered_endpoints.get(endpoint_template)
        if not meta:
            return None

        try:
            import httpx
            url = meta["sample_url"]
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(url, params=params)
                if resp.status_code == 200:
                    return resp.json()
        except Exception:
            pass
        return None

    # ═══════════════════════════════════════════
    #  İSTATİSTİKLER
    # ═══════════════════════════════════════════
    @property
    def captured(self) -> list[dict]:
        return self._captured[-100:]  # Son 100

    @property
    def discovered_endpoints(self) -> dict:
        return dict(self._discovered_endpoints)

    @property
    def stats(self) -> dict:
        return {
            **self._stats,
            "discovered_endpoints": len(self._discovered_endpoints),
            "ws_messages": len(self._ws_messages),
        }
