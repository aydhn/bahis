"""
evt_tail_scanner.py – Extreme Value Theory ile "Kara Kuğu" riski taraması.
Pareto dağılımı (GPD) ile uç olayları ve sürpriz potansiyelini hesaplar.
"""
from __future__ import annotations

import numpy as np
import polars as pl
from scipy import stats
from loguru import logger


class EVTTailScanner:
    """Aşırı değer teorisi ile sürpriz riski tarama motoru."""

    def __init__(self, threshold_quantile: float = 0.90):
        self._threshold_q = threshold_quantile
        logger.debug("EVTTailScanner başlatıldı.")

    def scan(self, features: pl.DataFrame) -> pl.DataFrame:
        """Her maç için kuyruk riski hesaplar."""
        results = []
        for row in features.iter_rows(named=True):
            mid = row.get("match_id", "")
            odds_vol = row.get("odds_volatility", 0.0)

            # İmplied probability spreads
            ho = row.get("home_odds", 2.5)
            do_ = row.get("draw_odds", 3.3)
            ao = row.get("away_odds", 3.0)

            implied = np.array([1/ho, 1/do_, 1/ao])
            implied /= implied.sum()

            # Sürpriz potansiyeli: düşük olasılıklı sonuç ne kadar muhtemel?
            min_prob = implied.min()
            surprise_factor = self._compute_surprise(implied, odds_vol)

            # GPD tail analizi (oran hareketleri üzerinden)
            tail_risk = self._gpd_analysis(implied, odds_vol)

            results.append({
                "match_id": mid,
                "tail_risk": tail_risk,
                "surprise_factor": surprise_factor,
                "min_implied_prob": float(min_prob),
                "black_swan_alert": tail_risk > 0.15,
            })

        return pl.DataFrame(results) if results else pl.DataFrame()

    def _compute_surprise(self, implied: np.ndarray, volatility: float) -> float:
        """Shannon entropi tabanlı sürpriz faktörü."""
        entropy = -np.sum(implied * np.log(implied + 1e-10))
        max_entropy = np.log(len(implied))
        normalized_entropy = entropy / max_entropy

        # Yüksek entropi + yüksek volatilite = yüksek sürpriz potansiyeli
        surprise = normalized_entropy * (1 + volatility * 5)
        return float(np.clip(surprise, 0, 1))

    def _gpd_analysis(self, implied: np.ndarray, volatility: float) -> float:
        """Generalized Pareto Distribution ile kuyruk riski."""
        try:
            if volatility <= 0:
                return float(implied.min() * 0.5)

            # Simülasyon: oran hareketlerini modellemek için
            np.random.seed(42)
            simulated_movements = np.random.normal(0, volatility, 500)
            simulated_movements = np.abs(simulated_movements)

            threshold = np.quantile(simulated_movements, self._threshold_q)
            exceedances = simulated_movements[simulated_movements > threshold] - threshold

            if len(exceedances) < 5:
                return float(implied.min() * 0.3)

            # GPD fit
            shape, loc, scale = stats.genpareto.fit(exceedances, floc=0)

            # P(X > 2*threshold) – aşırı hareket olasılığı
            tail_prob = stats.genpareto.sf(threshold, shape, loc=0, scale=scale)

            return float(np.clip(tail_prob, 0, 1))
        except Exception as e:
            logger.debug(f"GPD fit hatası: {e}")
            return float(implied.min() * 0.3)

    def assess_portfolio_tail(self, matches_data: list[dict]) -> dict:
        """Portföy genelinde sistemik kuyruk riski."""
        if not matches_data:
            return {"portfolio_tail_risk": 0.0, "alert": False}

        tail_risks = [m.get("tail_risk", 0.0) for m in matches_data]
        surprises = [m.get("surprise_factor", 0.0) for m in matches_data]

        # Korelasyon etkisi: bağımsız değillerse risk artar
        correlation_penalty = np.std(tail_risks) * len(tail_risks) * 0.1

        portfolio_risk = np.mean(tail_risks) + correlation_penalty
        portfolio_risk = float(np.clip(portfolio_risk, 0, 1))

        return {
            "portfolio_tail_risk": portfolio_risk,
            "max_surprise": float(max(surprises)) if surprises else 0.0,
            "alert": portfolio_risk > 0.2,
            "black_swans": sum(1 for t in tail_risks if t > 0.15),
        }
