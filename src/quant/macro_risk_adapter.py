"""
macro_risk_adapter.py – Makro-verileri Risk Yönetimine entegre eder.

Global kriz, enflasyon patlaması veya yüksek faiz oynaklığı durumlarında 
sistemin kasa yönetimini (Bankroll Management) dinamik olarak ayarlar.
"""
from loguru import logger
from typing import Any

class MacroRiskAdapter:
    def __init__(self, macro_scraper: Any):
        self.scraper = macro_scraper

    async def adjust_kelly(self, raw_kelly: float) -> float:
        """Kelly oranını makro-risk skoruna göre ayarlar."""
        risk_score = await self.scraper.get_macro_risk_score()
        
        # Örn: Risk skoru 0.8 ise (yüksek makro risk), bahsi %20 küçült.
        adjusted_kelly = raw_kelly * risk_score
        
        if risk_score < 0.7:
            logger.warning(f"[Risk:Macro] Yüksek küresel volatilite! Kelly {raw_kelly} -> {adjusted_kelly} seviyesine çekildi.")
            
        return round(adjusted_kelly, 4)

    def get_market_regime(self, vix: float) -> str:
        """Piyasa rejimini tanımlar."""
        if vix > 30: return "CRISIS"
        if vix > 20: return "VOLATILE"
        return "STABLE"
