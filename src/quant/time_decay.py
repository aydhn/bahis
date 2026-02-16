"""
time_decay.py – Logaritmik / Exponential Zaman Ağırlıklandırma + Takım Volatilite Endeksi.

3 yıl önceki veri ile geçen haftaki veri aynı ağırlıkta olamaz.
"Unutma Faktörü": W_t = e^(-λt)

Volatilite Endeksi: Takımların skor tutarsızlığını ölçer.
Yüksek vol → Gol bahsi öner, Düşük vol → Taraf bahsi öner.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import polars as pl
from loguru import logger


# ═══════════════════════════════════════════════
#  EXPONENTIAL TIME DECAY
# ═══════════════════════════════════════════════
class ExponentialTimeDecay:
    """Veriye zaman ağırlığı uygulayan modül.

    W_t = e^(-λ * t)
    t: gün cinsinden geçmiş süre
    λ: decay rate (yüksek → daha hızlı unutur)
    """

    PRESETS = {
        "aggressive": 0.01,    # Son 3 ay önemli
        "moderate": 0.005,     # Son 6 ay önemli (varsayılan)
        "conservative": 0.002, # Son 1.5 yıl önemli
        "ultra_slow": 0.001,   # Son 3 yıl önemli
    }

    def __init__(self, decay_rate: float = 0.005, preset: str | None = None):
        if preset and preset in self.PRESETS:
            self._lambda = self.PRESETS[preset]
        else:
            self._lambda = decay_rate

        self._half_life = math.log(2) / self._lambda if self._lambda > 0 else float("inf")
        logger.debug(
            f"TimeDecay başlatıldı: λ={self._lambda}, "
            f"yarı-ömür={self._half_life:.0f} gün"
        )

    @property
    def half_life_days(self) -> float:
        """Ağırlığın yarıya düştüğü gün sayısı."""
        return self._half_life

    def weight(self, days_ago: float) -> float:
        """Tek bir zaman noktası için ağırlık."""
        return math.exp(-self._lambda * max(days_ago, 0))

    def weight_from_date(self, match_date: datetime, reference: datetime | None = None) -> float:
        """Tarih nesnesinden ağırlık hesapla."""
        ref = reference or datetime.utcnow()
        days = (ref - match_date).total_seconds() / 86400
        return self.weight(days)

    def apply_weights(self, values: np.ndarray, days_ago: np.ndarray) -> np.ndarray:
        """Değerler dizisine zaman ağırlığı uygula."""
        weights = np.exp(-self._lambda * np.maximum(days_ago, 0))
        return values * weights

    def weighted_mean(self, values: np.ndarray, days_ago: np.ndarray) -> float:
        """Zaman ağırlıklı ortalama."""
        weights = np.exp(-self._lambda * np.maximum(days_ago, 0))
        total_w = weights.sum()
        if total_w < 1e-10:
            return float(np.mean(values))
        return float(np.sum(values * weights) / total_w)

    def weighted_std(self, values: np.ndarray, days_ago: np.ndarray) -> float:
        """Zaman ağırlıklı standart sapma."""
        w_mean = self.weighted_mean(values, days_ago)
        weights = np.exp(-self._lambda * np.maximum(days_ago, 0))
        total_w = weights.sum()
        if total_w < 1e-10:
            return float(np.std(values))
        variance = np.sum(weights * (values - w_mean) ** 2) / total_w
        return float(np.sqrt(max(variance, 0)))

    def apply_to_dataframe(self, df: pl.DataFrame,
                           date_col: str = "kickoff",
                           value_cols: list[str] | None = None) -> pl.DataFrame:
        """Polars DataFrame'e zaman ağırlığı sütunu ekler."""
        if df.is_empty() or date_col not in df.columns:
            return df

        now = datetime.utcnow()

        # days_ago hesapla
        try:
            dates = df[date_col].to_list()
            days_ago_list = []
            for d in dates:
                if isinstance(d, datetime):
                    days = (now - d).total_seconds() / 86400
                elif isinstance(d, str):
                    try:
                        dt = datetime.fromisoformat(d.replace("Z", "+00:00").replace("Z", ""))
                        days = (now - dt).total_seconds() / 86400
                    except (ValueError, TypeError):
                        days = 180
                else:
                    days = 180
                days_ago_list.append(max(days, 0))

            weights = [self.weight(d) for d in days_ago_list]
            df = df.with_columns(
                pl.Series("time_weight", weights),
                pl.Series("days_ago", days_ago_list),
            )
        except Exception as e:
            logger.debug(f"TimeDecay DataFrame hatası: {e}")

        return df

    def effective_sample_size(self, n_matches: int, avg_days_ago: float) -> float:
        """Ağırlıklar uygulandıktan sonraki efektif örneklem büyüklüğü."""
        weights = np.exp(-self._lambda * np.linspace(0, avg_days_ago * 2, n_matches))
        return float(weights.sum() ** 2 / (weights ** 2).sum())


# ═══════════════════════════════════════════════
#  TAKIM VOLATİLİTE ENDEKSİ (VIX)
# ═══════════════════════════════════════════════
@dataclass
class TeamVolProfile:
    """Takımın volatilite profili."""
    team: str
    volatility: float = 0.0         # Skor volatilitesi (std)
    goals_scored_std: float = 0.0   # Attığı gol std
    goals_conceded_std: float = 0.0 # Yediği gol std
    result_entropy: float = 0.0     # Sonuç entropisi (H/D/A)
    consistency: float = 0.0        # 1 - volatilite (tutarlılık)
    category: str = "medium"        # low / medium / high
    recommended_market: str = "1X2" # Önerilen pazar


class TeamVolatilityIndex:
    """Her takımın "VIX"ini hesaplayan modül.

    Düşük volatilite → 1-0, 2-1 biter → Taraf bahsi öner
    Yüksek volatilite → 5-0, 3-3 biter → Gol bahsi öner
    """

    VOL_THRESHOLDS = {"low": 0.8, "medium": 1.3}  # std sınırları

    def __init__(self, decay: ExponentialTimeDecay | None = None):
        self._decay = decay or ExponentialTimeDecay(preset="moderate")
        self._profiles: dict[str, TeamVolProfile] = {}
        logger.debug("TeamVolatilityIndex başlatıldı.")

    def calculate(self, team: str, match_history: list[dict]) -> TeamVolProfile:
        """Takımın volatilite profilini hesaplar.

        match_history: [{goals_for, goals_against, days_ago}, ...]
        """
        if not match_history:
            return TeamVolProfile(team=team)

        gf = np.array([m.get("goals_for", 0) for m in match_history], dtype=float)
        ga = np.array([m.get("goals_against", 0) for m in match_history], dtype=float)
        days = np.array([m.get("days_ago", 0) for m in match_history], dtype=float)
        total = gf + ga

        # Zaman ağırlıklı volatilite
        vol_total = self._decay.weighted_std(total, days)
        vol_gf = self._decay.weighted_std(gf, days)
        vol_ga = self._decay.weighted_std(ga, days)

        # Sonuç entropisi: H/D/A dağılımı ne kadar belirsiz?
        wins = sum(1 for m in match_history if m.get("goals_for", 0) > m.get("goals_against", 0))
        draws = sum(1 for m in match_history if m.get("goals_for", 0) == m.get("goals_against", 0))
        losses = len(match_history) - wins - draws
        n = max(len(match_history), 1)
        probs = np.array([wins/n, draws/n, losses/n]) + 1e-10
        entropy = -np.sum(probs * np.log(probs))
        max_entropy = np.log(3)
        norm_entropy = entropy / max_entropy

        # Kategori
        if vol_total < self.VOL_THRESHOLDS["low"]:
            category = "low"
            market = "1X2"
        elif vol_total < self.VOL_THRESHOLDS["medium"]:
            category = "medium"
            market = "1X2"
        else:
            category = "high"
            market = "goals"  # Alt/Üst, KG

        consistency = 1.0 - min(vol_total / 3.0, 1.0)

        profile = TeamVolProfile(
            team=team,
            volatility=float(vol_total),
            goals_scored_std=float(vol_gf),
            goals_conceded_std=float(vol_ga),
            result_entropy=float(norm_entropy),
            consistency=float(consistency),
            category=category,
            recommended_market=market,
        )

        self._profiles[team] = profile
        return profile

    def recommend_market(self, home: str, away: str) -> dict:
        """İki takımın volatilitesine göre bahis pazarı önerisi."""
        hp = self._profiles.get(home)
        ap = self._profiles.get(away)

        if not hp or not ap:
            return {"market": "1X2", "reason": "Yeterli volatilite verisi yok."}

        avg_vol = (hp.volatility + ap.volatility) / 2
        vol_diff = abs(hp.volatility - ap.volatility)

        if avg_vol > self.VOL_THRESHOLDS["medium"]:
            if hp.volatility > self.VOL_THRESHOLDS["medium"] and ap.volatility > self.VOL_THRESHOLDS["medium"]:
                return {
                    "market": "over_25",
                    "reason": f"Her iki takım yüksek volatilite "
                              f"(Ev:{hp.volatility:.2f}, Dep:{ap.volatility:.2f}). Üst 2.5 önerilir.",
                    "confidence": min(avg_vol / 2, 0.9),
                }
            else:
                return {
                    "market": "btts",
                    "reason": f"Volatilite farkı yüksek ({vol_diff:.2f}). KG Var önerilir.",
                    "confidence": 0.6,
                }
        elif avg_vol < self.VOL_THRESHOLDS["low"]:
            return {
                "market": "under_25",
                "reason": f"Her iki takım düşük volatilite "
                          f"(Ev:{hp.volatility:.2f}, Dep:{ap.volatility:.2f}). Alt 2.5 önerilir.",
                "confidence": min(hp.consistency * ap.consistency, 0.85),
            }
        else:
            # Medium volatilite → 1X2 uygundur
            if hp.consistency > ap.consistency:
                return {
                    "market": "1X2",
                    "reason": f"Ev sahibi daha tutarlı ({hp.consistency:.2f}). 1X2 önerilir.",
                    "confidence": 0.65,
                }
            else:
                return {
                    "market": "1X2",
                    "reason": "Orta volatilite – klasik 1X2 analizi uygun.",
                    "confidence": 0.55,
                }

    def get_profile(self, team: str) -> TeamVolProfile | None:
        return self._profiles.get(team)

    def all_profiles(self) -> list[dict]:
        return [
            {
                "team": p.team,
                "volatility": p.volatility,
                "category": p.category,
                "consistency": p.consistency,
                "recommended_market": p.recommended_market,
            }
            for p in sorted(self._profiles.values(), key=lambda x: x.volatility, reverse=True)
        ]
