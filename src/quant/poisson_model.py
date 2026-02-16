"""
poisson_model.py – Poisson dağılımı ile skor olasılıkları.
Futbol skorları nadir olaylardır (low-scoring events).
xG verileri üzerinden her olası skor çizgisinin yüzdesel olasılığını hesaplar.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import poisson
from itertools import product
from loguru import logger
import polars as pl


class PoissonModel:
    """Bivariate Poisson modeli ile skor olasılıkları."""

    def __init__(self, max_goals: int = 8):
        self._max_goals = max_goals
        logger.debug("PoissonModel başlatıldı.")

    def score_matrix(self, home_xg: float, away_xg: float) -> np.ndarray:
        """
        Skor olasılık matrisini döndürür.
        matrix[i][j] = P(Home=i, Away=j)
        """
        home_probs = poisson.pmf(range(self._max_goals + 1), mu=max(home_xg, 0.1))
        away_probs = poisson.pmf(range(self._max_goals + 1), mu=max(away_xg, 0.1))

        # Dış çarpım → bağımsız Poisson varsayımı
        matrix = np.outer(home_probs, away_probs)
        return matrix

    def match_outcome_probs(self, home_xg: float, away_xg: float) -> dict:
        """Maç sonucu olasılıklarını döndürür: H/D/A."""
        mat = self.score_matrix(home_xg, away_xg)
        n = mat.shape[0]

        p_home = sum(mat[i][j] for i in range(n) for j in range(n) if i > j)
        p_draw = sum(mat[i][i] for i in range(n))
        p_away = sum(mat[i][j] for i in range(n) for j in range(n) if i < j)

        return {
            "prob_home": float(p_home),
            "prob_draw": float(p_draw),
            "prob_away": float(p_away),
        }

    def over_under_probs(self, home_xg: float, away_xg: float) -> dict:
        """Alt/Üst gol olasılıklarını hesaplar."""
        mat = self.score_matrix(home_xg, away_xg)
        n = mat.shape[0]

        thresholds = [0.5, 1.5, 2.5, 3.5, 4.5]
        result = {}
        for thresh in thresholds:
            over = sum(mat[i][j] for i in range(n) for j in range(n) if i + j > thresh)
            under = 1 - over
            key = str(thresh).replace(".", "")
            result[f"over_{key}"] = float(over)
            result[f"under_{key}"] = float(under)

        return result

    def btts_probs(self, home_xg: float, away_xg: float) -> dict:
        """Karşılıklı Gol (BTTS) olasılığı."""
        mat = self.score_matrix(home_xg, away_xg)
        n = mat.shape[0]

        btts_yes = sum(mat[i][j] for i in range(1, n) for j in range(1, n))
        return {
            "btts_yes": float(btts_yes),
            "btts_no": float(1 - btts_yes),
        }

    def most_likely_scores(self, home_xg: float, away_xg: float, top_n: int = 10) -> list[dict]:
        """En olası skorları sıralı döndürür."""
        mat = self.score_matrix(home_xg, away_xg)
        n = mat.shape[0]

        scores = []
        for i in range(n):
            for j in range(n):
                scores.append({
                    "score": f"{i}-{j}",
                    "home_goals": i,
                    "away_goals": j,
                    "probability": float(mat[i][j]),
                })

        scores.sort(key=lambda x: x["probability"], reverse=True)
        return scores[:top_n]

    def correct_score_value(self, home_xg: float, away_xg: float, odds_map: dict) -> list[dict]:
        """Doğru skor pazarında value olan bahisleri bulur.
        odds_map = {"1-0": 6.5, "2-1": 8.0, ...}
        """
        mat = self.score_matrix(home_xg, away_xg)
        values = []

        for score_str, odds in odds_map.items():
            parts = score_str.split("-")
            if len(parts) == 2:
                h, a = int(parts[0]), int(parts[1])
                if h < mat.shape[0] and a < mat.shape[1]:
                    prob = mat[h][a]
                    ev = prob * odds - 1
                    values.append({
                        "score": score_str,
                        "probability": float(prob),
                        "odds": odds,
                        "ev": float(ev),
                        "is_value": ev > 0.02,
                    })

        values.sort(key=lambda x: x["ev"], reverse=True)
        return values

    def predict_for_dataframe(self, features: pl.DataFrame) -> pl.DataFrame:
        """DataFrame üzerinden toplu Poisson tahmini."""
        results = []
        for row in features.iter_rows(named=True):
            mid = row.get("match_id", "")
            home_xg = row.get("home_xg", 1.3) or 1.3
            away_xg = row.get("away_xg", 1.1) or 1.1

            outcome = self.match_outcome_probs(home_xg, away_xg)
            over_under = self.over_under_probs(home_xg, away_xg)
            btts = self.btts_probs(home_xg, away_xg)
            top_scores = self.most_likely_scores(home_xg, away_xg, 3)

            result = {
                "match_id": mid,
                "poisson_home": outcome["prob_home"],
                "poisson_draw": outcome["prob_draw"],
                "poisson_away": outcome["prob_away"],
                "poisson_over25": over_under.get("over_25", 0.5),
                "poisson_under25": over_under.get("under_25", 0.5),
                "poisson_btts_yes": btts["btts_yes"],
                "poisson_expected_goals": home_xg + away_xg,
                "poisson_most_likely": top_scores[0]["score"] if top_scores else "1-1",
            }
            results.append(result)

        return pl.DataFrame(results) if results else pl.DataFrame()
