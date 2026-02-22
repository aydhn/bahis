"""
cross_sport_risk_manager.py – Branşlar Arası Portföy ve Risk Yönetimi.

Sistemin çoklu branş (Futbol, Basketbol, Tenis) sinyalleri arasındaki 
riski dengelemesini sağlar. Diversifikasyon kuralı uygulanır.
"""
import numpy as np
from loguru import logger
from typing import Dict, Any, List

class CrossSportRiskManager:
    def __init__(self, db: Any = None):
        self.db = db
        # Branş bazlı risk limitleri (Maksimum % sermaye)
        self.sport_limits = {
            "football": 0.40,
            "basketball": 0.30,
            "tennis": 0.30
        }

    def calculate_correlation(self, sport_a: str, sport_b: str) -> float:
        """İki branş arasındaki istatistiksel bağımlılığı ölçer (Asset Correlation)."""
        # Genelde bahis branşları bağımsızdır (~0), 
        # ancak aynı ligler/oyuncular üzerinden korelasyon olabilir.
        return 0.05 # Mock korelasyon

    def optimize_allocation(self, signals: List[dict], total_capital: float) -> List[dict]:
        """Sinyalleri toplam portföy riskine göre normalize eder."""
        sport_exposure = {"football": 0, "basketball": 0, "tennis": 0}
        
        for sig in signals:
            sport = sig.get("sport", "football")
            limit = self.sport_limits.get(sport, 0.20) * total_capital
            
            # Exposure kontrolü
            if sport_exposure[sport] < limit:
                # Kelly oranını korelasyona göre düzelt (Diversification Discount)
                sig["kelly_final"] = sig.get("kelly_raw", 0.02) * 0.9
                sport_exposure[sport] += sig["kelly_final"] * total_capital
            else:
                sig["kelly_final"] = 0.0
                sig["reason"] = f"Max {sport} exposure reached."
                
        logger.info(f"[Risk] Portföy Dağılımı: {sport_exposure}")
        return signals
