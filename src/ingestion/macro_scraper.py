"""
macro_scraper.py – Küresel ekonomik takvim ve makro-ekonomik veri takipçisi.

Investing.com veya Yahoo Finance üzerinden faiz kararları, enflasyon verileri 
ve VIX endeksi gibi yüksek etkili (High Impact) olayları çeker.
"""
import asyncio
import httpx
from loguru import logger
from datetime import datetime
from typing import List, Dict

class MacroScraper:
    def __init__(self):
        self._calendar_url = "https://query1.finance.yahoo.com/v7/finance/options/^VIX" # Örnek VIX takip
        self._headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

    async def fetch_vix(self) -> float:
        """VIX (Volatilite) endeksini çeker."""
        try:
            async with httpx.AsyncClient(headers=self._headers) as client:
                # Yahoo Finance API (v7) üzerinden VIX meta verisi
                resp = await client.get("https://query1.finance.yahoo.com/v8/finance/chart/^VIX?interval=1d")
                if resp.status_code == 200:
                    data = resp.json()
                    vix_val = data["chart"]["result"][0]["meta"]["regularMarketPrice"]
                    logger.info(f"[Macro] Güncel VIX Endeksi: {vix_val}")
                    return float(vix_val)
        except Exception as e:
            logger.error(f"[Macro] VIX çekme hatası: {e}")
        return 20.0 # Default/Normal volatilite

    async def fetch_economic_events(self) -> List[Dict]:
        """Bugünün kritik ekonomik olaylarını (Faiz, CPI vb.) listeler."""
        # Şimdilik simüle edilmiş, ileride Investing.com scraper eklenebilir.
        return [
            {"event": "FED Interest Rate Decision", "impact": "High", "date": datetime.now().strftime("%Y-%m-%d")},
            {"event": "US CPI Data", "impact": "High", "date": datetime.now().strftime("%Y-%m-%d")}
        ]

    async def get_macro_risk_score(self) -> float:
        """Makro-ekonomik verilere göre bir risk katsayısı (0.5 - 1.5) üretir."""
        vix = await self.fetch_vix()
        events = await self.fetch_economic_events()
        
        # VIX > 30 ise panik/yüksek risk (Stake düşür)
        # VIX < 15 ise düşük risk/güven (Stake artırabilir)
        base_score = 1.0
        if vix > 30: base_score *= 0.7
        if vix > 40: base_score *= 0.5
        if vix < 15: base_score *= 1.2
        
        # Bugün yüksek etkili olay varsa riski ihtiyati olarak %20 azalt
        high_impact = any(e["impact"] == "High" for e in events)
        if high_impact:
            base_score *= 0.8
            
        logger.info(f"[Macro] Risk Katsayısı: {base_score:.2f}")
        return round(base_score, 2)
