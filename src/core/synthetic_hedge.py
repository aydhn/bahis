"""
synthetic_hedge.py – Sentetik Risk Koruma (Hedging) Motoru.

Profesyonel portföy yönetimi mantığıyla çalışır. Eğer sistem belirli bir takım 
veya lige çok fazla 'exposure' (risk maruziyeti) verdiyse, 
bu modül diğer piyasalarda (Asian Handicap, Alt/Üst) zıt pozisyonlar arayarak 
'sentetik sigorta' oluşturur.
"""
from typing import List, Dict, Any, Optional
from loguru import logger

class SyntheticHedgingEngine:
    def __init__(self, db: Any = None, max_exposure_per_group: float = 0.10):
        self.db = db
        self.max_exposure = max_exposure_per_group # Tek bir takım/lig için max risk %10

    def calculate_exposure(self, active_bets: List[Dict]) -> Dict[str, float]:
        """Mevcut bahislerdeki risk dağılımını (exposure) hesaplar."""
        exposure = {}
        for bet in active_bets:
            group = bet.get("team_id") or bet.get("league_id")
            stake = bet.get("stake", 0.0)
            exposure[group] = exposure.get(group, 0.0) + stake
        return exposure

    async def find_hedge_opportunities(self, exposure: Dict[str, float], market_prices: List[Dict]) -> List[Dict]:
        """
        Riskli gruplar için hedging fırsatları arar.
        Örn: A Takımına çok bahis yapıldıysa, A Takımının rakibine handicap+1.5 arar.
        """
        hedges = []
        for group, amount in exposure.items():
            if amount > self.max_exposure:
                logger.info(f"[Hedging] {group} için yüksek risk ({amount:.2f}). Koruma aranıyor...")
                
                # Bu grup için piyasa fiyatlarını filtrele
                options = [m for m in market_prices if m.get("team_id") == group or m.get("related_group") == group]
                
                # Zıt (inverse) sonuçları bul
                for opt in options:
                    if opt.get("type") == "inverse":
                        hedges.append({
                            "type": "SYNTHETIC_HEDGE",
                            "group": group,
                            "market": opt["market"],
                            "odds": opt["odds"],
                            "required_stake": (amount - self.max_exposure) * 0.8 # Risk fazlasının bir kısmını kapa
                        })
        return hedges

    async def run_batch(self, active_bets: List[Dict], live_odds: List[Dict], **kwargs) -> List[Dict]:
        """Orchestrator üzerinden çağrılan hedging döngüsü."""
        exposure = self.calculate_exposure(active_bets)
        return await self.find_hedge_opportunities(exposure, live_odds)
