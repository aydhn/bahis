"""
scenario_simulator.py – Monte Carlo Engine for Match Outcomes.

Provides deep statistical insight by simulating a match 10,000 times
using the underlying probability distributions (Poisson/Negative Binomial).
Used to calculate Value at Risk (VaR) and Probability of Ruin.
"""
from typing import Dict, Any, List
import numpy as np
from loguru import logger
from src.core.rust_engine import RustEngine

class ScenarioSimulator:
    """
    Simulates match outcomes to assess risk and variance.
    Uses RustEngine for high-performance Monte Carlo simulations if available.
    """

    def __init__(self, n_sims: int = 10000):
        self.n_sims = n_sims
        self.rust_engine = RustEngine()

    def simulate_match(self,
                       home_xg: float,
                       away_xg: float,
                       match_id: str = "") -> Dict[str, Any]:
        """
        Runs Monte Carlo simulation for a single match based on xG.
        Returns outcome distribution.
        """
        # Accelerate with Rust if possible
        if self.rust_engine.engine_name in ["rust", "numba"]:
            res = self.rust_engine.monte_carlo_sim(
                n_sims=self.n_sims,
                home_xg=home_xg,
                away_xg=away_xg
            )
            # Rust engine returns aggregated stats directly
            # We need distributions for histogram, but Rust impl only returns stats for speed
            # If we need detailed histograms, we might need a different Rust method or fallback
            # For now, we simulate simplified distributions in Python for histogram ONLY if needed
            # Or accept that histogram is based on stats (Poisson approximation)

            # Reconstruct basic distribution for histogram from means (Poisson)
            # This is fast enough for visualization
            home_goals = np.random.poisson(home_xg, min(self.n_sims, 1000)) # Smaller sample for hist
            away_goals = np.random.poisson(away_xg, min(self.n_sims, 1000))

            return {
                "match_id": match_id,
                "prob_home": res["home_win"],
                "prob_draw": res["draw"],
                "prob_away": res["away_win"],
                "avg_goals": res["mean_goals"],
                "sim_count": self.n_sims,
                "home_goals_dist": self._condense_dist(home_goals, n=len(home_goals)),
                "away_goals_dist": self._condense_dist(away_goals, n=len(away_goals)),
                "engine": self.rust_engine.engine_name
            }

        # Fallback to standard Numpy
        home_goals = np.random.poisson(home_xg, self.n_sims)
        away_goals = np.random.poisson(away_xg, self.n_sims)

        # Determine outcomes
        home_wins = np.sum(home_goals > away_goals)
        draws = np.sum(home_goals == away_goals)
        away_wins = np.sum(away_goals > home_goals)

        # Probabilities
        p_home = home_wins / self.n_sims
        p_draw = draws / self.n_sims
        p_away = away_wins / self.n_sims

        # Goal Distributions (for ASCII histogram)
        total_goals = home_goals + away_goals
        avg_goals = np.mean(total_goals)

        # Calculate VaR (Value at Risk) if we bet on Home
        # This requires odds. Simulation gives us probabilities.

        return {
            "match_id": match_id,
            "prob_home": p_home,
            "prob_draw": p_draw,
            "prob_away": p_away,
            "avg_goals": avg_goals,
            "sim_count": self.n_sims,
            "home_goals_dist": self._condense_dist(home_goals),
            "away_goals_dist": self._condense_dist(away_goals),
            "engine": "numpy"
        }

    def simulate_bet_risk(self,
                          prob_win: float,
                          odds: float,
                          stake: float) -> Dict[str, Any]:
        """
        Simulates the PnL distribution of a specific bet over repeated trials.
        Useful to understand variance.
        """
        # Simulate N outcomes (Bernoulli trial)
        outcomes = np.random.random(self.n_sims) < prob_win

        # Calculate PnL for each sim
        pnl = np.where(outcomes, stake * (odds - 1), -stake)

        expected_value = np.mean(pnl)
        var_95 = np.percentile(pnl, 5) # 5th percentile (Downside risk)

        # Probability of losing money (if we repeated this bet)
        # For a single bet, it's just 1-prob_win.
        # But this is useful for a portfolio of identical bets.

        return {
            "ev": expected_value,
            "var_95": var_95,
            "max_drawdown_sim": np.min(np.cumsum(pnl)) # Worst path if sequential
        }

    def _condense_dist(self, data: np.ndarray, n: int = None) -> Dict[int, float]:
        """Helper to create a frequency map for histograms."""
        if n is None: n = self.n_sims
        unique, counts = np.unique(data, return_counts=True)
        return dict(zip(unique.tolist(), (counts / n).tolist()))

    def generate_ascii_histogram(self, dist: Dict[int, float], label: str) -> str:
        """Generates a text-based histogram for Telegram."""
        lines = [f"📊 {label} Distribution:"]
        sorted_keys = sorted(dist.keys())
        for k in sorted_keys:
            if k > 5: continue # Clip tail
            val = dist[k]
            bar_len = int(val * 20)
            bar = "█" * bar_len
            lines.append(f"{k}: {bar} {val:.1%}")
        return "\n".join(lines)
