"""
benter_model.py – Bill Benter Tarzı Gelişmiş Olasılık Modeli.

Bu modül, klasik Poisson modelini "Benter Adjustments" ile geliştirir.
Benter'ın başarısının sırrı, saf istatistiksel çıktıyı (Poisson)
niteliksel faktörlerle (Sakatlık, Hava Durumu, Motivasyon) birleştirmesiydi.

Yetenekler:
  - Contextual xG Adjustment: Hava, yorgunluk, eksik oyuncu etkileri.
  - Zero-Inflation Correction: 0-0 skorunu Poisson'un azımsamasını düzeltir.
  - Value Bet Detection: "Margin of Safety" ile EV hesabı.
"""
from __future__ import annotations

import numpy as np
from loguru import logger
from src.quant.finance.liquidity_engine import LiquidityEngine
from src.quant.models.poisson_model import PoissonModel

class BenterModel(PoissonModel):
    """
    Bill Benter tarzı hibrit model.
    Baz model: Poisson.
    Katmanlar: Expert Adjustment + Zero-Inflation + Kelly Edge.
    """

    def __init__(self, max_goals: int = 8):
        super().__init__(max_goals)
        self.liquidity_engine = LiquidityEngine()
        logger.info("BenterModel başlatıldı: Expert Adjustments ve Liquidity simülasyonu aktif.")

    def apply_contextual_adjustments(self, home_xg: float, away_xg: float, context: dict) -> tuple[float, float]:
        """
        Saf xG değerlerini maçın hikayesine göre düzeltir.

        Context parametreleri (0-1 arası etki katsayıları veya boolean):
        - fatigue_home: Ev sahibi yorgunluk (0.0 - 1.0)
        - fatigue_away: Deplasman yorgunluk
        - rain: Yağmur var mı? (bool) -> Gol beklentisini düşürür.
        - missing_key_player_home: Eksik kilit oyuncu (bool) -> xG düşürür.
        - motivation_high_home: Derbi/Şampiyonluk maçı (bool) -> xG artırır.
        """
        adj_home = home_xg
        adj_away = away_xg

        # 1. Yorgunluk Etkisi (Her %10 yorgunluk -> %2 performans kaybı)
        if "fatigue_home" in context:
            f = context["fatigue_home"]
            adj_home *= (1.0 - (f * 0.2)) # Max %20 kayıp

        if "fatigue_away" in context:
            f = context["fatigue_away"]
            adj_away *= (1.0 - (f * 0.2))

        # 2. Hava Durumu (Yağmur/Kar -> Düşük skor)
        if context.get("rain", False):
            adj_home *= 0.90
            adj_away *= 0.90

        # 3. Eksik Oyuncu (Kilit oyuncu yoksa %15 kayıp)
        if context.get("missing_key_player_home", False):
            adj_home *= 0.85
        if context.get("missing_key_player_away", False):
            adj_away *= 0.85

        # 4. Motivasyon (Derbi -> %10 artış)
        if context.get("motivation_high_home", False):
            adj_home *= 1.10

        return round(adj_home, 2), round(adj_away, 2)

    def zero_inflated_correction(self, matrix: np.ndarray, correction_factor: float = 0.05) -> np.ndarray:
        """
        0-0 skor olasılığını (matrix[0][0]) yapay olarak artırır.
        Futbolda 0-0, Poisson dağılımının öngördüğünden daha sık olur.

        Args:
            matrix: Olasılık matrisi (Home x Away)
            correction_factor: 0-0 olasılığına eklenecek mutlak puan (örn 0.05 = %5).
                               Diğer skorlardan orantılı düşülür.
        """
        if correction_factor <= 0:
            return matrix

        current_prob = matrix[0][0]
        target_prob = min(current_prob + correction_factor, 1.0)

        # Farkı diğer tüm hücrelerden orantılı çıkar
        # (1 - current_prob) toplam olasılıktan 'diff' kadar çalacağız.
        scaling_factor = (1.0 - target_prob) / (1.0 - current_prob)

        new_matrix = matrix * scaling_factor
        new_matrix[0][0] = target_prob

        return new_matrix

    def calculate_benter_probabilities(self, home_xg: float, away_xg: float, context: dict | None = None) -> dict:
        """
        Tüm Benter düzeltmelerini uygulayarak final olasılıkları döndürür.
        """
        ctx = context or {}

        # 1. Adjust Inputs
        h_adj, a_adj = self.apply_contextual_adjustments(home_xg, away_xg, ctx)

        # 2. Base Matrix
        mat = self.score_matrix(h_adj, a_adj)

        # 3. Zero-Inflation (Eğer defansif bir maç bekleniyorsa)
        if h_adj + a_adj < 2.5:
             mat = self.zero_inflated_correction(mat, correction_factor=0.03)

        # 4. Calculate Outcomes
        n = mat.shape[0]
        p_home = float(sum(mat[i][j] for i in range(n) for j in range(n) if i > j))
        p_draw = float(sum(mat[i][i] for i in range(n)))
        p_away = float(sum(mat[i][j] for i in range(n) for j in range(n) if i < j))

        return {
            "prob_home": p_home,
            "prob_draw": p_draw,
            "prob_away": p_away,
            "adjusted_xg_home": h_adj,
            "adjusted_xg_away": a_adj,
            "matrix_00": float(mat[0][0])
        }

    def detect_value_bet(self, outcomes: dict, odds: dict, min_edge: float = 0.05, fractional_kelly: float = 0.5, league: str = "Default") -> list[dict]:
        """
        Bahis fırsatlarını (Value Bets) tespit eder.
        Bill Benter tarzı: Liquidity slippage hesaplayarak dinamik Kelly kesri uygular.
        """
        bets = []

        # Dinamik min_edge ayarı (Belirsizlik varsa edge beklentisini artır)
        uncertainty = outcomes.get("epistemic_uncertainty", 0.0)
        adjusted_min_edge = min_edge + (uncertainty * 0.1)  # Max %10 ek güvenlik marjı

        for key, selection, prob_key in [("1", "HOME", "prob_home"), ("X", "DRAW", "prob_draw"), ("2", "AWAY", "prob_away")]:
            if key in odds:
                p = outcomes.get(prob_key, 0.0)
                o = odds[key]
                edge = p * o - 1

                # Check execution slippage on a standard sizing
                # Simulate a standard 100 unit bet to see if edge holds
                exec_price, slippage_pct = self.liquidity_engine.simulate_execution(stake=100.0, odds=o, league=league)
                real_edge = (p * exec_price) - 1.0

                if real_edge > adjusted_min_edge:
                    # True Kelly: (Edge / (Odds - 1))
                    k_frac = (real_edge / (exec_price - 1.0)) if exec_price > 1.0 else 0.0

                    # Benter refinement: Dynamically adjust the fractional kelly based on real_edge depth
                    # If real edge is huge (e.g., >10%), push closer to Full Kelly.
                    dynamic_kelly = fractional_kelly
                    if real_edge > 0.10:
                        dynamic_kelly = min(0.8, fractional_kelly + (real_edge - 0.10) * 2)

                    suggested_frac = k_frac * dynamic_kelly

                    bets.append({
                        "selection": selection,
                        "probability": p,
                        "odds": o,
                        "exec_price": exec_price,
                        "slippage_pct": slippage_pct,
                        "edge": edge,
                        "real_edge": real_edge,
                        "kelly_fraction": k_frac,
                        "suggested_stake_fraction": suggested_frac
                    })

        bets.sort(key=lambda x: x["real_edge"], reverse=True)
        return bets
