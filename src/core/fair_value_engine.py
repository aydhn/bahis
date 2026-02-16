"""
fair_value_engine.py – Sentetik Oran Üretimi (Synthetic Odds / Fair Value).

Bahisçi gibi değil, bahis bürosu gibi düşünmek.
Kendi "Adil Oran" (Fair Odds) üretip, piyasa ile kıyaslayarak
matematiksel hataları (inefficiency) bulmak.

İşleyiş:
  1. Ensemble modelden gelen olasılığı al (P_model = 0.60)
  2. Fair Odds = 1 / P_model = 1.667
  3. Market Odds = 1.90 (bahis şirketinin sunduğu)
  4. Value Edge = (Market / Fair) - 1 = +14.0%
  5. Edge > 0 → OYNA, Edge < 0 → OYNAMA

Ayrıca:
  - Margin-free fair odds hesaplama (Pinnacle marjı çıkarma)
  - Implied probability extraction
  - Sharp bookmaker vs Soft bookmaker karşılaştırma
  - No-vig odds hesaplama
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from loguru import logger


@dataclass
class FairValueResult:
    """Tek bir bahis için Fair Value analizi."""
    match_id: str = ""
    selection: str = ""           # home / draw / away / over25 / btts
    # Model tahmini
    model_prob: float = 0.0       # Ensemble modelinin olasılığı
    model_confidence: float = 0.0
    # Adil oran
    fair_odds: float = 0.0        # 1 / model_prob
    # Piyasa oranı
    market_odds: float = 0.0
    market_prob: float = 0.0      # 1 / market_odds (marjlı)
    # Marjsız piyasa olasılığı
    novig_prob: float = 0.0       # No-vig hesaplama sonrası
    novig_odds: float = 0.0
    # Value analizi
    value_edge: float = 0.0       # (market / fair) - 1
    value_edge_pct: str = ""
    is_value: bool = False
    # Kelly stake
    kelly_stake: float = 0.0
    kelly_fraction: float = 0.25  # Fractional Kelly
    # Sınıflandırma
    tier: str = ""                # "premium" / "standard" / "marginal" / "no_value"


class FairValueEngine:
    """Sentetik Oran Üretimi ve Value Edge Tespiti.

    Kullanım:
        engine = FairValueEngine()

        # Tek maç analizi
        result = engine.analyze(
            model_prob=0.60,
            market_odds=1.90,
            selection="home",
            match_id="gs_fb",
        )
        print(result.value_edge)  # +0.14 (+14%)
        print(result.tier)        # "premium"

        # Toplu analiz
        batch = engine.analyze_batch(signals, market_odds)
    """

    # Value Edge sınıflandırma eşikleri
    TIERS = {
        "premium": 0.10,    # >10% edge → yüksek güven
        "standard": 0.05,   # 5-10% → normal value
        "marginal": 0.02,   # 2-5% → düşük güven
    }

    def __init__(self, min_edge: float = 0.02, kelly_fraction: float = 0.25,
                 max_odds: float = 15.0, min_prob: float = 0.05):
        self._min_edge = min_edge
        self._kelly_fraction = kelly_fraction
        self._max_odds = max_odds
        self._min_prob = min_prob
        self._history: list[FairValueResult] = []
        logger.debug(
            f"FairValueEngine: min_edge={min_edge:.0%}, "
            f"kelly={kelly_fraction:.0%}"
        )

    # ═══════════════════════════════════════════
    #  TEK BAHİS ANALİZİ
    # ═══════════════════════════════════════════
    def analyze(self, model_prob: float, market_odds: float,
                selection: str = "", match_id: str = "",
                confidence: float = 0.5) -> FairValueResult:
        """Tek bir bahis için fair value analizi."""

        # Güvenlik kontrolleri
        model_prob = max(self._min_prob, min(model_prob, 0.99))
        market_odds = max(1.01, min(market_odds, self._max_odds))

        # Fair odds: model olasılığının doğrudan karşılığı
        fair_odds = 1.0 / model_prob

        # Piyasa implied probability (marjlı)
        market_prob = 1.0 / market_odds

        # Value Edge: piyasanın "fazla" verdiği miktar
        value_edge = (market_odds / fair_odds) - 1.0

        # Sınıflandırma
        if value_edge >= self.TIERS["premium"]:
            tier = "premium"
        elif value_edge >= self.TIERS["standard"]:
            tier = "standard"
        elif value_edge >= self.TIERS["marginal"]:
            tier = "marginal"
        else:
            tier = "no_value"

        # Kelly Criterion
        kelly = self._kelly_stake(model_prob, market_odds)

        result = FairValueResult(
            match_id=match_id,
            selection=selection,
            model_prob=model_prob,
            model_confidence=confidence,
            fair_odds=round(fair_odds, 3),
            market_odds=market_odds,
            market_prob=round(market_prob, 4),
            value_edge=round(value_edge, 4),
            value_edge_pct=f"{value_edge:+.1%}",
            is_value=value_edge >= self._min_edge,
            kelly_stake=round(kelly, 4),
            kelly_fraction=self._kelly_fraction,
            tier=tier,
        )

        self._history.append(result)
        return result

    # ═══════════════════════════════════════════
    #  TOPLU ANALİZ
    # ═══════════════════════════════════════════
    def analyze_batch(self, signals: list[dict],
                      market_odds: dict[str, dict] | None = None
                      ) -> list[FairValueResult]:
        """Birden fazla sinyal için toplu fair value analizi.

        signals: [{"match_id": "...", "selection": "home", "prob": 0.60}, ...]
        market_odds: {"match_id": {"home": 1.90, "draw": 3.40, "away": 4.20}}
        """
        results = []
        for sig in signals:
            mid = sig.get("match_id", "")
            sel = sig.get("selection", "home")
            prob = sig.get("prob", sig.get("model_prob", sig.get("prob_home", 0.33)))

            # Piyasa oranını bul
            m_odds = 0.0
            if market_odds and mid in market_odds:
                m_odds = market_odds[mid].get(sel, 0)
            if not m_odds:
                m_odds = sig.get("market_odds", sig.get("odds", 0))

            if prob > 0 and m_odds > 1.0:
                result = self.analyze(
                    model_prob=prob,
                    market_odds=m_odds,
                    selection=sel,
                    match_id=mid,
                    confidence=sig.get("confidence", 0.5),
                )
                results.append(result)

        # Value'ye göre sırala
        results.sort(key=lambda r: r.value_edge, reverse=True)
        return results

    # ═══════════════════════════════════════════
    #  NO-VIG ODDS (Marjsız Oran Hesaplama)
    # ═══════════════════════════════════════════
    def remove_vig(self, home_odds: float, draw_odds: float,
                   away_odds: float) -> dict:
        """Bahisçinin marjını çıkararak gerçek olasılıkları hesapla.

        Yöntem: Multiplicative / Power method
        """
        if any(o <= 1.0 for o in (home_odds, draw_odds, away_odds)):
            return {}

        # Ham implied probabilities
        ip_home = 1.0 / home_odds
        ip_draw = 1.0 / draw_odds
        ip_away = 1.0 / away_odds
        overround = ip_home + ip_draw + ip_away  # >1.0 = marj

        # Basit normalizasyon (additive)
        nv_home = ip_home / overround
        nv_draw = ip_draw / overround
        nv_away = ip_away / overround

        # Shin's method (daha doğru, özellikle büyük marjlarda)
        shin_probs = self._shin_method(home_odds, draw_odds, away_odds)

        return {
            "raw_margin": round((overround - 1) * 100, 2),  # Marj %
            "overround": round(overround, 4),
            # Additive no-vig
            "novig_home": round(nv_home, 4),
            "novig_draw": round(nv_draw, 4),
            "novig_away": round(nv_away, 4),
            "novig_home_odds": round(1 / nv_home, 3),
            "novig_draw_odds": round(1 / nv_draw, 3),
            "novig_away_odds": round(1 / nv_away, 3),
            # Shin's method
            "shin_home": round(shin_probs.get("home", nv_home), 4),
            "shin_draw": round(shin_probs.get("draw", nv_draw), 4),
            "shin_away": round(shin_probs.get("away", nv_away), 4),
        }

    @staticmethod
    def _shin_method(home_odds: float, draw_odds: float,
                      away_odds: float) -> dict:
        """Shin's method: insider trading'i hesaba katan marj çıkarma.

        Daha doğru özellikle favori/underdog ayrımında.
        """
        odds = [home_odds, draw_odds, away_odds]
        n = len(odds)
        implied = [1.0 / o for o in odds]
        total = sum(implied)

        # Shin'in z parametresi (Newton-Raphson ile çöz)
        z = (total - 1) / (n - 1 + total)  # Başlangıç tahmini

        for _ in range(50):
            numerator = sum(
                (np.sqrt(z**2 + 4 * (1 - z) * (ip**2 / total)) - z)
                for ip in implied
            ) / (2 * (1 - z)) - 1

            if abs(numerator) < 1e-10:
                break

            # Gradient (yaklaşık)
            h = 1e-8
            z_h = z + h
            f_h = sum(
                (np.sqrt(z_h**2 + 4 * (1 - z_h) * (ip**2 / total)) - z_h)
                for ip in implied
            ) / (2 * (1 - z_h)) - 1

            derivative = (f_h - numerator) / h
            if abs(derivative) > 1e-12:
                z -= numerator / derivative
                z = max(0, min(z, 0.5))

        # Shin probabilities
        probs = {}
        labels = ["home", "draw", "away"]
        for i, (ip, label) in enumerate(zip(implied, labels)):
            p = (np.sqrt(z**2 + 4 * (1 - z) * (ip**2 / total)) - z) / (2 * (1 - z))
            probs[label] = float(max(0.01, min(p, 0.99)))

        return probs

    # ═══════════════════════════════════════════
    #  SHARP vs SOFT BOOKMAKER
    # ═══════════════════════════════════════════
    def compare_bookmakers(self, pinnacle_odds: dict,
                            soft_odds: dict) -> dict:
        """Sharp (Pinnacle) vs Soft bookmaker karşılaştırması.

        Pinnacle oranları "gerçek olasılığa" en yakın kabul edilir.
        Soft bookmaker daha yüksek oran veriyorsa → value.
        """
        pin_novig = self.remove_vig(
            pinnacle_odds.get("home", 0),
            pinnacle_odds.get("draw", 0),
            pinnacle_odds.get("away", 0),
        )

        results = {}
        for sel in ("home", "draw", "away"):
            pin_prob = pin_novig.get(f"novig_{sel}", 0)
            soft_o = soft_odds.get(sel, 0)

            if pin_prob > 0 and soft_o > 1.0:
                edge = (soft_o * pin_prob) - 1
                results[sel] = {
                    "pinnacle_prob": pin_prob,
                    "soft_odds": soft_o,
                    "edge": round(edge, 4),
                    "edge_pct": f"{edge:+.1%}",
                    "is_value": edge > 0,
                }

        return results

    # ═══════════════════════════════════════════
    #  KELLY CRITERION
    # ═══════════════════════════════════════════
    def _kelly_stake(self, prob: float, odds: float) -> float:
        """Optimal Kelly stake hesapla.

        Kelly% = (p * o - 1) / (o - 1) * fraction
        """
        if odds <= 1.0 or prob <= 0:
            return 0.0

        kelly = (prob * odds - 1) / (odds - 1)
        kelly *= self._kelly_fraction  # Fractional Kelly

        return max(kelly, 0.0)

    # ═══════════════════════════════════════════
    #  İSTATİSTİKLER
    # ═══════════════════════════════════════════
    def summary(self) -> dict:
        """Tarihsel fair value analiz özeti."""
        if not self._history:
            return {"n": 0}

        values = [r for r in self._history if r.is_value]
        edges = [r.value_edge for r in self._history]

        return {
            "total_analyzed": len(self._history),
            "value_found": len(values),
            "value_pct": len(values) / max(len(self._history), 1),
            "avg_edge": float(np.mean(edges)) if edges else 0,
            "max_edge": float(max(edges)) if edges else 0,
            "tier_distribution": {
                "premium": sum(1 for r in self._history if r.tier == "premium"),
                "standard": sum(1 for r in self._history if r.tier == "standard"),
                "marginal": sum(1 for r in self._history if r.tier == "marginal"),
                "no_value": sum(1 for r in self._history if r.tier == "no_value"),
            },
            "avg_kelly": float(np.mean([r.kelly_stake for r in values])) if values else 0,
        }
