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

        # Precompute masks for vectorization
        n = self._max_goals + 1
        i_idx, j_idx = np.indices((n, n))

        self._mask_home = np.tril(np.ones((n, n)), k=-1)
        self._mask_draw = np.eye(n)
        self._mask_away = np.triu(np.ones((n, n)), k=1)

        self._mask_over25 = (i_idx + j_idx > 2.5)
        self._mask_under25 = ~self._mask_over25

        self._mask_btts_yes = (i_idx > 0) & (j_idx > 0)

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
        """DataFrame üzerinden toplu Poisson tahmini (Vektörize)."""
        if features.is_empty():
            return pl.DataFrame()

        # Extract columns with defaults
        # Handle 'home_xg' missing column case gracefully if needed,
        # but usually features DF has it.
        try:
            home_xg_col = features.get_column("home_xg")
        except Exception:
            home_xg_col = pl.Series("home_xg", [None] * len(features))

        try:
            away_xg_col = features.get_column("away_xg")
        except Exception:
            away_xg_col = pl.Series("away_xg", [None] * len(features))

        # Fill nulls
        home_xg = home_xg_col.fill_null(1.3).to_numpy()
        away_xg = away_xg_col.fill_null(1.1).to_numpy()

        # Handle 0.0 values (which behave like falsy in 'or' logic of original code)
        # Original: row.get("home_xg", 1.3) or 1.3 -> if 0.0, becomes 1.3
        # Create writeable copies or use where to ensure we don't hit read-only errors
        home_xg = np.where(home_xg == 0, 1.3, home_xg)
        away_xg = np.where(away_xg == 0, 1.1, away_xg)

        # Apply min threshold
        home_xg = np.maximum(home_xg, 0.1)
        away_xg = np.maximum(away_xg, 0.1)

        # --- Vectorized Calculation ---
        n = len(home_xg)
        goals = np.arange(self._max_goals + 1)

        # Poisson PMF broadcasting
        # home_xg: (N,) -> (N, 1)
        # goals: (G,) -> (1, G)
        # Result: (N, G)
        home_probs = poisson.pmf(goals[None, :], mu=home_xg[:, None])
        away_probs = poisson.pmf(goals[None, :], mu=away_xg[:, None])

        # Outer product: (N, G, 1) * (N, 1, G) -> (N, G, G)
        matrix = home_probs[:, :, None] * away_probs[:, None, :]

        # Outcome Probabilities
        p_home = np.sum(matrix * self._mask_home, axis=(1, 2))
        p_draw = np.sum(matrix * self._mask_draw, axis=(1, 2))
        p_away = np.sum(matrix * self._mask_away, axis=(1, 2))

        # Over/Under 2.5
        p_over25 = np.sum(matrix * self._mask_over25, axis=(1, 2))
        p_under25 = np.sum(matrix * self._mask_under25, axis=(1, 2))

        # BTTS
        p_btts_yes = np.sum(matrix * self._mask_btts_yes, axis=(1, 2))

        # Most likely score
        # Flatten last two dimensions: (N, G*G)
        n_goals = self._max_goals + 1
        matrix_flat = matrix.reshape(n, -1)
        top_idx = np.argmax(matrix_flat, axis=1)

        top_i = top_idx // n_goals
        top_j = top_idx % n_goals

        # Vectorized string formatting is tricky in numpy, using list comprehension is fast enough for strings
        most_likely_scores = [f"{i}-{j}" for i, j in zip(top_i, top_j)]

        # Construct Result DataFrame
        match_ids = features.get_column("match_id") if "match_id" in features.columns else pl.Series("match_id", [""] * n)

        return pl.DataFrame({
            "match_id": match_ids,
            "poisson_home": p_home,
            "poisson_draw": p_draw,
            "poisson_away": p_away,
            "poisson_over25": p_over25,
            "poisson_under25": p_under25,
            "poisson_btts_yes": p_btts_yes,
            "poisson_expected_goals": home_xg + away_xg,
            "poisson_most_likely": most_likely_scores,
        })
