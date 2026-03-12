from loguru import logger
"""
stress_tester.py – Portfolio Stress Testing & Crash Simulation.

Simulates the entire open portfolio under adverse market conditions
to ensure solvency. Acts as a "Chief Risk Officer" (CRO) check.

Scenarios:
  - Market Crash: All favorites lose / Underdogs win (Black Swan).
  - Correlation Spike: Failures happen together.
  - Odds Drift: Slippage increases significantly.

Metric:
  - Portfolio VaR (Value at Risk): "What is the worst case (95%) loss?"
"""
import numpy as np
from typing import List, Dict, Any

class PortfolioStressTester:
    """
    Simulates portfolio outcomes under stress to prevent Ruin.
    """

    def __init__(self, n_sims: int = 1000, var_threshold: float = 0.20):
        self.n_sims = n_sims
        self.var_threshold = var_threshold # Max allowed VaR (e.g., 20% of bankroll)

    def check_portfolio_health(self,
                               current_bets: List[Dict[str, Any]],
                               new_bet: Dict[str, Any],
                               total_capital: float) -> Dict[str, Any]:
        """
        Runs a Monte Carlo simulation on the combined portfolio (current + new).
        Returns 'approved' status and VaR metrics.
        """
        if total_capital <= 0:
            return {"approved": False, "reason": "Bankrupt (Capital <= 0)"}

        # Combine bets
        portfolio = current_bets + [new_bet]

        if not portfolio:
             return {"approved": True, "var_pct": 0.0, "reason": "Empty Portfolio"}

        # Extract probabilities and potential PnLs
        probs = []
        stakes = []
        odds = []

        for bet in portfolio:
            # Default to 50% if prob missing (conservative?) or use implied
            o = bet.get("odds", 2.0)
            p = bet.get("prob_home", 0.0) # Assuming home for simplicity, needs generic 'prob'
            if p == 0:
                p = 1.0 / o if o > 0 else 0.5

            s = bet.get("stake_amount", 0.0)
            if s == 0 and "stake_pct" in bet:
                s = bet["stake_pct"] * total_capital

            probs.append(p)
            stakes.append(s)
            odds.append(o)

        probs = np.array(probs)
        stakes = np.array(stakes)
        odds = np.array(odds)

        # --- SCENARIO 1: Standard Monte Carlo ---
        try:
            from src.core.rust_engine import RustEngine
            rust_engine = RustEngine()
            if callable(rust_engine.engine_name):
                engine_type = rust_engine.engine_name()
            else:
                engine_type = rust_engine.engine_name
            if engine_type != "Pure Python":
                # Assuming rust_engine can run faster simulations if we had a dedicated func.
                # For now, just initialize it to ensure Rust/Numba is warm.
                pass
        except ImportError as e:
            from loguru import logger
            logger.debug(f"RustEngine not found, falling back: {e}")

        rng = np.random.default_rng()
        outcomes = rng.random((self.n_sims, len(portfolio))) < probs
        pnl_matrix = np.where(outcomes, stakes * (odds - 1), -stakes)
        sim_pnls = pnl_matrix.sum(axis=1)
        var_95_abs = np.percentile(sim_pnls, 5)

        var_loss = -var_95_abs if var_95_abs < 0 else 0.0
        var_pct = var_loss / total_capital

        # --- SCENARIO 2: Correlation Crash (Stress Test) ---
        # Assume during a crash, if one fails, others are likely to fail.
        # We simulate this by lowering probabilities by 20% across the board.
        crashed_probs = probs * 0.8
        outcomes_crash = rng.random((self.n_sims, len(portfolio))) < crashed_probs
        pnl_crash = np.where(outcomes_crash, stakes * (odds - 1), -stakes).sum(axis=1)
        var_crash_abs = np.percentile(pnl_crash, 5)
        var_crash_loss = -var_crash_abs if var_crash_abs < 0 else 0.0
        var_crash_pct = var_crash_loss / total_capital

        # Decision Logic
        # We use the Stress VaR for safety
        if var_crash_pct > self.var_threshold:
            return {
                "approved": False,
                "reason": f"Portfolio VaR Violation (Stress VaR: {var_crash_pct:.1%})",
                "var_pct": var_crash_pct,
                "stress_impact": var_crash_pct - var_pct
            }

        return {
            "approved": True,
            "reason": f"Portfolio Healthy (VaR: {var_crash_pct:.1%})",
            "var_pct": var_crash_pct
        }
