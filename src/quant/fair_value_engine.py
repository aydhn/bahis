"""
fair_value_engine.py – Pazar Değer Hizalama ve Etik Bahis Motoru.

Bill Benter ve Edward Thorp gibi "Quant" üstadlarının felsefesini taşır:
"Sadece matematiksel avantajın (Edge) olduğu değil, pazarın 'yanlış' 
fiyatladığı durumlarda aksiyon al."

Bu modül, botun pazar verimliliğine (Efficiency) bakarak bahislerin
'Adil Değer' (Fair Value) ile uyumunu denetler.
"""
from loguru import logger

class FairValueEngine:
    def __init__(self, db_manager=None, min_value_threshold: float = 0.05):
        self.db = db_manager
        self.min_value = min_value_threshold
        logger.info("FairValueEngine initialized.")

    def is_aligned_with_value(self, model_prob: float, market_odds: float) -> bool:
        """
        Olasılık ve oran dengesini kontrol eder.
        Edge = (Prob * Odds) - 1
        """
        edge = (model_prob * market_odds) - 1
        return edge >= self.min_value

    def filter_value_bets(self, candidates: list[dict]) -> list[dict]:
        """
        Sinyalleri 'Fair Value' süzgecinden geçirir.
        Sadece gerçek değer barındıranları (noise olmayanlar) döndürür.
        """
        value_bets = [b for b in candidates if self.is_aligned_with_value(b['prob'], b['odds'])]
        logger.info(f"FairValue Filtering: {len(candidates)} -> {len(value_bets)} value bets.")
        return value_bets

    def adjust_risk_profile(self, volatility_index: float) -> float:
        """
        JP Morgan Risk Profili: Yüksek volatilite anlarında risk iştahını
        (Kelly fraction) dinamik olarak düşürür.
        """
        if volatility_index > 0.8: # Çok riskli
            return 0.2
        elif volatility_index > 0.5: # Hareketli
            return 0.5
        return 1.0 # Normal
