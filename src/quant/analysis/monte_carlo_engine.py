"""
monte_carlo_engine.py – 10.000 simülasyonla ampirik maç tahmini.
"Bu maçta %60 ihtimalle 2.5 üst olur" demek yerine:
"10.000 simülasyonun 6.200'ünde 3 gol ve üzeri atıldı" diyen ampirik yaklaşım.
"""
from __future__ import annotations

import numpy as np
import polars as pl
from loguru import logger


class MonteCarloEngine:
    """Monte Carlo ile maç simülasyonu motoru."""

    def __init__(self, n_simulations: int = 10_000, seed: int | None = 42):
        self._n_sim = n_simulations
        self._rng = np.random.default_rng(seed)
        logger.debug(f"MonteCarloEngine başlatıldı – {n_simulations:,} simülasyon.")

    def simulate_match(self, home_xg: float, away_xg: float) -> dict:
        """Tek bir maçı n_sim kere simüle eder."""
        home_xg = max(home_xg, 0.1)
        away_xg = max(away_xg, 0.1)

        # Poisson dağılımından gol sayıları çek
        home_goals = self._rng.poisson(home_xg, self._n_sim)
        away_goals = self._rng.poisson(away_xg, self._n_sim)
        total_goals = home_goals + away_goals

        # Sonuçlar
        home_wins = np.sum(home_goals > away_goals)
        draws = np.sum(home_goals == away_goals)
        away_wins = np.sum(home_goals < away_goals)

        # Over/Under
        over_15 = np.sum(total_goals > 1.5)
        over_25 = np.sum(total_goals > 2.5)
        over_35 = np.sum(total_goals > 3.5)

        # BTTS
        btts = np.sum((home_goals > 0) & (away_goals > 0))

        # Skor dağılımı
        # Vectorized optimization for score counting (approx 7x faster)
        # Assuming max goals < 100 per team, which is safe for football
        combined = home_goals * 100 + away_goals
        unique, counts = np.unique(combined, return_counts=True)

        # Sort by counts descending and take top 10
        sorted_indices = np.argsort(-counts)[:10]

        top_scores = []
        for idx in sorted_indices:
            code = unique[idx]
            count = counts[idx]
            h = code // 100
            a = code % 100
            top_scores.append((f"{h}-{a}", int(count)))

        return {
            "n_simulations": self._n_sim,
            "home_win_count": int(home_wins),
            "draw_count": int(draws),
            "away_win_count": int(away_wins),
            "prob_home": float(home_wins / self._n_sim),
            "prob_draw": float(draws / self._n_sim),
            "prob_away": float(away_wins / self._n_sim),
            "over_15_count": int(over_15),
            "over_25_count": int(over_25),
            "over_35_count": int(over_35),
            "prob_over_15": float(over_15 / self._n_sim),
            "prob_over_25": float(over_25 / self._n_sim),
            "prob_over_35": float(over_35 / self._n_sim),
            "btts_count": int(btts),
            "prob_btts": float(btts / self._n_sim),
            "avg_total_goals": float(np.mean(total_goals)),
            "avg_home_goals": float(np.mean(home_goals)),
            "avg_away_goals": float(np.mean(away_goals)),
            "std_total_goals": float(np.std(total_goals)),
            "top_scores": [{"score": s, "count": c, "pct": c / self._n_sim} for s, c in top_scores],
        }

    def simulate_with_correlation(self, home_xg: float, away_xg: float, rho: float = 0.1) -> dict:
        """Korelasyonlu bivaryat Poisson simülasyonu.
        rho > 0: açık maç (çok gol), rho < 0: defansif maç.
        """
        home_xg = max(home_xg, 0.1)
        away_xg = max(away_xg, 0.1)

        # Dixon-Coles düzeltmesi ile korelasyon ekleme
        lambda_0 = max(rho, 0)

        home_independent = self._rng.poisson(max(home_xg - lambda_0, 0.05), self._n_sim)
        away_independent = self._rng.poisson(max(away_xg - lambda_0, 0.05), self._n_sim)
        common = self._rng.poisson(lambda_0, self._n_sim)

        home_goals = home_independent + common
        away_goals = away_independent + common
        total_goals = home_goals + away_goals

        home_wins = np.sum(home_goals > away_goals)
        draws = np.sum(home_goals == away_goals)
        away_wins = np.sum(home_goals < away_goals)

        return {
            "n_simulations": self._n_sim,
            "correlation": rho,
            "prob_home": float(home_wins / self._n_sim),
            "prob_draw": float(draws / self._n_sim),
            "prob_away": float(away_wins / self._n_sim),
            "prob_over_25": float(np.sum(total_goals > 2.5) / self._n_sim),
            "prob_btts": float(np.sum((home_goals > 0) & (away_goals > 0)) / self._n_sim),
            "avg_total_goals": float(np.mean(total_goals)),
        }

    def simulate_season(self, matches: list[dict], n_seasons: int = 1000) -> dict:
        """Bir sezonu n kere simüle ederek şampiyonluk olasılıklarını hesaplar."""
        team_points = {}
        for _ in range(n_seasons):
            season_points = {}
            for match in matches:
                home = match.get("home_team", "")
                away = match.get("away_team", "")
                home_xg = match.get("home_xg", 1.3)
                away_xg = match.get("away_xg", 1.1)

                hg = self._rng.poisson(home_xg)
                ag = self._rng.poisson(away_xg)

                season_points.setdefault(home, 0)
                season_points.setdefault(away, 0)

                if hg > ag:
                    season_points[home] += 3
                elif hg == ag:
                    season_points[home] += 1
                    season_points[away] += 1
                else:
                    season_points[away] += 3

            # Şampiyon
            if season_points:
                champion = max(season_points, key=season_points.get)
                team_points[champion] = team_points.get(champion, 0) + 1

        # Olasılıklar
        probs = {team: count / n_seasons for team, count in team_points.items()}
        probs = dict(sorted(probs.items(), key=lambda x: x[1], reverse=True))

        return {"n_seasons": n_seasons, "champion_probabilities": probs}

    def predict_bulk(self, home_xg: np.ndarray, away_xg: np.ndarray) -> dict:
        """Vectorized simulation for multiple matches."""
        n_matches = len(home_xg)
        # Ensure minimum xG
        home_xg = np.maximum(home_xg, 0.1)
        away_xg = np.maximum(away_xg, 0.1)

        # (n_matches, n_sim)
        # Generate all simulations at once
        h_sims = self._rng.poisson(home_xg[:, None], size=(n_matches, self._n_sim))
        a_sims = self._rng.poisson(away_xg[:, None], size=(n_matches, self._n_sim))

        home_wins = np.sum(h_sims > a_sims, axis=1)
        draws = np.sum(h_sims == a_sims, axis=1)
        away_wins = np.sum(h_sims < a_sims, axis=1)

        total_goals = h_sims + a_sims
        over_25 = np.sum(total_goals > 2.5, axis=1)
        btts = np.sum((h_sims > 0) & (a_sims > 0), axis=1)
        avg_goals = np.mean(total_goals, axis=1)

        return {
            "prob_home": home_wins / self._n_sim,
            "prob_draw": draws / self._n_sim,
            "prob_away": away_wins / self._n_sim,
            "prob_over_25": over_25 / self._n_sim,
            "prob_btts": btts / self._n_sim,
            "avg_total_goals": avg_goals
        }

    def predict_for_dataframe(self, features: pl.DataFrame) -> pl.DataFrame:
        """DataFrame üzerinden toplu MC simülasyonu (Vectorized)."""
        if features.is_empty():
            return pl.DataFrame()

        # Handle missing columns or nulls with defaults
        home_xg = features.get_column("home_xg").fill_null(1.3).to_numpy() if "home_xg" in features.columns else np.full(features.height, 1.3)
        away_xg = features.get_column("away_xg").fill_null(1.1).to_numpy() if "away_xg" in features.columns else np.full(features.height, 1.1)
        match_ids = features.get_column("match_id").to_list() if "match_id" in features.columns else [""] * features.height

        # Use vectorized bulk prediction
        results = self.predict_bulk(home_xg, away_xg)

        return pl.DataFrame({
            "match_id": match_ids,
            "mc_home": results["prob_home"],
            "mc_draw": results["prob_draw"],
            "mc_away": results["prob_away"],
            "mc_over25": results["prob_over_25"],
            "mc_btts": results["prob_btts"],
            "mc_avg_goals": results["avg_total_goals"],
            "mc_simulations": np.full(features.height, self._n_sim, dtype=int)
        })
