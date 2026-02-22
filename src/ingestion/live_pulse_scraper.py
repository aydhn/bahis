"""
live_pulse_scraper.py – Canlı Maç ve Oran Takip Motoru.

Playwright kullanarak canlı maçların dakikasını, skorunu ve 
oran değişimlerini anlık olarak yakalar.
"""
import asyncio
from typing import List, Dict, Any
from loguru import logger
from datetime import datetime

class LivePulseScraper:
    def __init__(self, headless: bool = True):
        self.headless = headless
        self._browser = None
        self._pw = None

    async def start(self):
        """Browser başlat."""
        from playwright.async_api import async_playwright
        self._pw = await async_playwright().start()
        self._browser = await self._pw.chromium.launch(headless=self.headless)
        logger.info("[LivePulse] Browser başlatıldı.")

    async def fetch_live_scores(self) -> List[Dict[str, Any]]:
        """Canlı skorları ve dakikaları çekir."""
        if not self._browser:
            await self.start()
        
        page = await self._browser.new_page()
        live_data = []
        try:
            # Flashscore Canlı Sekmesi (Örnek URL)
            await page.goto("https://www.flashscore.com.tr/", wait_until="networkidle", timeout=30000)
            
            # Canlı butonuna bas (eğer gerekliyse)
            # await page.click(".filters__tab--live")
            
            rows = await page.query_selector_all(".event__match--live")
            for row in rows:
                match_id = await row.get_attribute("id")
                home = await (await row.query_selector(".event__homeParticipant")).inner_text()
                away = await (await row.query_selector(".event__awayParticipant")).inner_text()
                score_home = await (await row.query_selector(".event__score--home")).inner_text()
                score_away = await (await row.query_selector(".event__score--away")).inner_text()
                minute = await (await row.query_selector(".event__stage--live")).inner_text()
                
                live_data.append({
                    "id": match_id,
                    "home": home.strip(),
                    "away": away.strip(),
                    "score": f"{score_home}-{score_away}",
                    "minute": minute.strip(),
                    "timestamp": datetime.now().isoformat()
                })
        except Exception as e:
            logger.error(f"[LivePulse] Veri çekme hatası: {e}")
        finally:
            await page.close()
        
        return live_data

    async def stop(self):
        if self._browser:
            await self._browser.close()
        if self._pw:
            await self._pw.stop()
