"""
evt_tail_scanner.py – Extreme Value Theory ile "Kara Kuğu" riski taraması.

Pareto dağılımı (GPD) ile uç olayları ve sürpriz potansiyelini hesaplar.
Peaks-over-threshold (POT) yöntemiyle kuyruk dağılımını modelleyerek
"imkansız" sürprizlerin gerçekleşme olasılığını matematiksel olarak ölçer.

Kavramlar:
  - Extreme Value Theory (EVT): Nadir olayların istatistiksel modellenmesi
  - Generalized Pareto Distribution (GPD): Eşik aşımlarının dağılımı
  - Shannon Entropy: Olasılık dağılımının belirsizlik ölçüsü
  - Tail Index (ξ): Kuyruğun kalınlığı — ξ > 0 "kalın kuyruk" (tehlikeli)
  - Exceedance: Eşiğin üzerindeki gözlemler
  - Portfolio Tail Dependence: Maçlar arası kuyruk korelasyonu

Akış:
  1. Her maç için oran verisinden implied probability hesapla
  2. Shannon entropi ile sürpriz potansiyelini ölç
  3. GPD ile kuyruk riski modelini fit et
  4. Portföy genelinde sistemik kuyruk riski hesapla
  5. Black Swan uyarısı (tail_risk > %15) tetikle
"""
from __future__ import annotations

import math
import numpy as np
import polars as pl
from scipy import stats
from loguru import logger


def _safe_float(value, default: float = 0.0) -> float:
    """Polars null, None, NaN ve geçersiz değerleri güvenli float'a çevirir."""
    if value is None:
        return default
    try:
        f = float(value)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


class EVTTailScanner:
    """Aşırı değer teorisi ile sürpriz riski tarama motoru."""

    def __init__(self, threshold_quantile: float = 0.90):
        self._threshold_q = threshold_quantile
        self._scan_count = 0
        self._total_black_swans = 0
        logger.debug(f"EVTTailScanner başlatıldı (threshold_q={threshold_quantile})")

    def scan(self, features: pl.DataFrame) -> pl.DataFrame:
        """Her maç için kuyruk riski hesaplar.

        Args:
            features: match_id, home_odds, draw_odds, away_odds, odds_volatility
                      sütunlarını içeren Polars DataFrame.

        Returns:
            match_id, tail_risk, surprise_factor, min_implied_prob,
            black_swan_alert, entropy, tail_index sütunlu DataFrame.
        """
        self._scan_count += 1
        results = []

        for row in features.iter_rows(named=True):
            try:
                mid = row.get("match_id", "")
                odds_vol = _safe_float(row.get("odds_volatility"), 0.0)

                ho = _safe_float(row.get("home_odds"), 0.0)
                do_ = _safe_float(row.get("draw_odds"), 0.0)
                ao = _safe_float(row.get("away_odds"), 0.0)

                if ho < 1.01:
                    ho = 2.5
                if do_ < 1.01:
                    do_ = 3.3
                if ao < 1.01:
                    ao = 3.0

                implied = np.array([1.0 / ho, 1.0 / do_, 1.0 / ao])
                total = implied.sum()
                if total <= 0:
                    implied = np.array([0.4, 0.3, 0.3])
                else:
                    implied /= total

                min_prob = float(implied.min())
                surprise = self._compute_surprise(implied, odds_vol)
                tail = self._gpd_analysis(implied, odds_vol)
                entropy = self._shannon_entropy(implied)
                tail_idx = self._estimate_tail_index(implied, odds_vol)

                is_swan = tail > 0.15
                if is_swan:
                    self._total_black_swans += 1

                results.append({
                    "match_id": mid,
                    "tail_risk": round(tail, 6),
                    "surprise_factor": round(surprise, 6),
                    "min_implied_prob": round(min_prob, 6),
                    "black_swan_alert": is_swan,
                    "entropy": round(entropy, 6),
                    "tail_index": round(tail_idx, 6),
                })

            except Exception as e:
                logger.debug(f"[EVT] Satır hatası ({row.get('match_id','?')}): {e}")
                results.append({
                    "match_id": row.get("match_id", "error"),
                    "tail_risk": 0.0,
                    "surprise_factor": 0.0,
                    "min_implied_prob": 0.33,
                    "black_swan_alert": False,
                    "entropy": 0.0,
                    "tail_index": 0.0,
                })

        return pl.DataFrame(results) if results else pl.DataFrame()

    def _shannon_entropy(self, implied: np.ndarray) -> float:
        """Shannon entropy — dağılımın belirsizlik ölçüsü."""
        p = np.clip(implied, 1e-10, 1.0)
        return float(-np.sum(p * np.log(p)))

    def _compute_surprise(self, implied: np.ndarray, volatility: float) -> float:
        """Shannon entropi + volatilite bazlı sürpriz faktörü.

        Yüksek entropi (eşit dağılım) + yüksek volatilite = yüksek sürpriz.
        """
        entropy = self._shannon_entropy(implied)
        max_entropy = np.log(len(implied))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

        vol_boost = 1.0 + min(volatility, 1.0) * 5.0
        surprise = normalized_entropy * vol_boost
        return float(np.clip(surprise, 0, 1))

    def _gpd_analysis(self, implied: np.ndarray, volatility: float) -> float:
        """Generalized Pareto Distribution ile kuyruk riski.

        Peaks-over-threshold: eşiğin üzerindeki gözlemleri GPD ile modelleyerek
        aşırı hareket olasılığını hesaplar.
        """
        try:
            if volatility <= 0:
                return float(np.clip(implied.min() * 0.5, 0, 1))

            rng = np.random.default_rng(42)
            n_sim = 1000
            simulated = np.abs(rng.normal(0, max(volatility, 0.001), n_sim))

            threshold = np.quantile(simulated, self._threshold_q)
            exceedances = simulated[simulated > threshold] - threshold

            if len(exceedances) < 5:
                return float(np.clip(implied.min() * 0.3, 0, 1))

            shape, loc, scale = stats.genpareto.fit(exceedances, floc=0)

            tail_prob = stats.genpareto.sf(threshold, shape, loc=0, scale=scale)

            kurtosis_penalty = 0.0
            if shape > 0:
                kurtosis_penalty = min(shape * 0.1, 0.2)

            final_risk = float(np.clip(tail_prob + kurtosis_penalty, 0, 1))
            return final_risk

        except Exception as e:
            logger.debug(f"GPD fit hatası: {e}")
            return float(np.clip(implied.min() * 0.3, 0, 1))

    def _estimate_tail_index(self, implied: np.ndarray, volatility: float) -> float:
        """Hill estimator ile tail index tahmini.

        ξ > 0: Kalın kuyruk (Pareto-tipi, tehlikeli)
        ξ = 0: Exponential kuyruk (orta risk)
        ξ < 0: İnce kuyruk (Weibull-tipi, güvenli)
        """
        try:
            if volatility <= 0:
                return 0.0
            rng = np.random.default_rng(42)
            data = np.abs(rng.normal(0, max(volatility, 0.001), 500))
            data = np.sort(data)[::-1]
            k = max(int(len(data) * 0.1), 5)
            top_k = data[:k]
            threshold = data[k]
            if threshold <= 0:
                return 0.0
            hill = float(np.mean(np.log(top_k / threshold)))
            return round(hill, 4)
        except Exception:
            return 0.0

    def assess_portfolio_tail(self, matches_data: list[dict]) -> dict:
        """Portföy genelinde sistemik kuyruk riski.

        Maçlar arasındaki kuyruk bağımlılığını (tail dependence)
        hesaplayarak portföy çapında sistemik risk ölçer.
        """
        if not matches_data:
            return {"portfolio_tail_risk": 0.0, "alert": False,
                    "max_surprise": 0.0, "black_swans": 0,
                    "concentration_risk": 0.0}

        tail_risks = [_safe_float(m.get("tail_risk"), 0.0) for m in matches_data]
        surprises = [_safe_float(m.get("surprise_factor"), 0.0) for m in matches_data]

        n = len(tail_risks)
        mean_risk = float(np.mean(tail_risks))
        std_risk = float(np.std(tail_risks)) if n > 1 else 0.0

        correlation_penalty = std_risk * math.sqrt(n) * 0.05

        concentration = float(np.max(tail_risks) / max(sum(tail_risks), 1e-10))

        portfolio_risk = mean_risk + correlation_penalty + concentration * 0.05
        portfolio_risk = float(np.clip(portfolio_risk, 0, 1))

        return {
            "portfolio_tail_risk": round(portfolio_risk, 6),
            "max_surprise": round(float(max(surprises)), 6) if surprises else 0.0,
            "alert": portfolio_risk > 0.2,
            "black_swans": sum(1 for t in tail_risks if t > 0.15),
            "concentration_risk": round(concentration, 4),
            "mean_tail_risk": round(mean_risk, 6),
        }

    def get_stats(self) -> dict:
        """Tarayıcı istatistikleri."""
        return {
            "scan_count": self._scan_count,
            "total_black_swans": self._total_black_swans,
            "threshold_quantile": self._threshold_q,
        }
