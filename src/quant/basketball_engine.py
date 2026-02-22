"""
basketball_engine.py – Basketbol için Skellam Dağılımı ve Handikap Modelleme.

Basketbol skorları arasındaki farkı (point spread) modellemek için 
iki bağımsız Poisson dağılımının farkı olan Skellam dağılımını kullanır.
"""
import numpy as np
from scipy.stats import skellam, poisson
from loguru import logger
from typing import Dict, Any, List

class BasketballEngine:
    def __init__(self):
        pass

    def predict_spread(self, home_lambda: float, away_lambda: float, spread: float) -> float:
        """Belirli bir handikapın (spread) tutma olasılığını hesaplar."""
        # home - away > spread olasılığı
        # Skellam(mu1, mu2) -> k = home - away
        # P(K > spread) = 1 - cdf(spread)
        prob = skellam.sf(spread, home_lambda, away_lambda)
        return float(prob)

    def predict_totals(self, home_lambda: float, away_lambda: float, total: float) -> float:
        """Toplam sayı (Over/Under) olasılığını hesaplar."""
        # Toplam sayı iki bağımsız Poisson'un toplamı olan Poisson(mu1 + mu2) dağılımına uyar
        total_lambda = home_lambda + away_lambda
        prob_over = 1 - poisson.cdf(total, total_lambda)
        return float(prob_over)

    def process_match(self, match_data: dict) -> dict:
        """Basketbol maç verisini işler."""
        home_exp = 110.5 # Örnek beklenen sayı
        away_exp = 108.2
        
        spread_val = -2.5 # Örnek handikap
        prob_spread = self.predict_spread(home_exp, away_exp, spread_val)
        
        return {
            "match_id": match_data.get("id"),
            "sport": "basketball",
            "p_home_spread": round(prob_spread, 4),
            "p_over_215": round(self.predict_totals(home_exp, away_exp, 215.5), 4),
            "recommendation": "NBA Home Spread" if prob_spread > 0.65 else "No Bet"
        }
