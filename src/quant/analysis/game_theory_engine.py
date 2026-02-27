"""
game_theory_engine.py – Game Theoretic Analysis (Nash Equilibrium).

Models the betting market as a 2-player zero-sum game between the Bettor (Hero)
and the Bookmaker (Villain). It calculates optimal mixed strategies (randomization)
to prevent exploitation in adversarial markets.

Concepts:
  - Zero-Sum Game: Bettor's gain is Bookie's loss (simplified).
  - Nash Equilibrium: A state where neither player can improve by changing strategy alone.
  - Minimax: Minimizing the maximum possible loss.
  - Mixed Strategy: Playing actions with specific probabilities to remain unexploitable.

Usage:
    engine = GameTheoryEngine()
    # Payoff matrix (Bettor's view): Rows=Bettor Actions, Cols=Bookie Actions
    # E.g., Bettor: [Bet Home, Bet Away, Pass], Bookie: [Shift Odds Home, Shift Odds Away, Hold]
    payoff_matrix = np.array([[...]])
    strategy, value = engine.solve_nash(payoff_matrix)
"""
from __future__ import annotations

import numpy as np
from loguru import logger
from dataclasses import dataclass

try:
    from scipy.optimize import linprog
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False
    logger.warning("scipy not found – GameTheoryEngine limited to basic heuristics.")

@dataclass
class GameResult:
    optimal_strategy: np.ndarray  # Probability distribution over actions
    game_value: float             # Expected payoff at equilibrium
    is_solved: bool = False
    method: str = "none"

class GameTheoryEngine:
    """
    Solves 2-player zero-sum games to find optimal betting strategies.
    """

    def __init__(self):
        logger.debug("GameTheoryEngine initialized.")

    def solve_nash(self, payoff_matrix: np.ndarray) -> GameResult:
        """
        Solves for the row player's optimal mixed strategy (Nash Equilibrium)
        using Linear Programming.

        Args:
            payoff_matrix: (M, N) matrix where A[i, j] is payoff to row player
                           when row plays i and col plays j.

        Returns:
            GameResult containing probability vector and game value.
        """
        if not SCIPY_OK:
            return self._heuristic_solution(payoff_matrix)

        M, N = payoff_matrix.shape

        # We want to maximize v subject to:
        # Sum(p_i * A_ij) >= v  for all j=1..N
        # Sum(p_i) = 1
        # p_i >= 0
        #
        # Equivalent to minimizing -v subject to:
        # -Sum(p_i * A_ij) + v <= 0
        # Sum(p_i) = 1
        # p_i >= 0
        #
        # Variables for linprog: x = [p_0, ..., p_{M-1}, v] (size M+1)
        # Objective: minimize -v -> c = [0, ..., 0, -1]

        c = np.zeros(M + 1)
        c[-1] = -1.0

        # Inequality constraints: -Sum(p_i * A_ij) + v <= 0
        # We have N constraints (one for each column/opponent action)
        # Format: A_ub @ x <= b_ub
        # A_ub row k: [-A_0k, -A_1k, ..., -A_{M-1}k, 1]

        A_ub = np.zeros((N, M + 1))
        # Transpose of payoff matrix (negated) goes into first M columns
        A_ub[:, :M] = -payoff_matrix.T
        A_ub[:, M] = 1.0  # Coefficient for v

        b_ub = np.zeros(N)

        # Equality constraint: Sum(p_i) = 1
        # A_eq @ x = b_eq
        # Row: [1, 1, ..., 1, 0]
        A_eq = np.zeros((1, M + 1))
        A_eq[0, :M] = 1.0
        b_eq = np.array([1.0])

        # Bounds: p_i >= 0, v is unbounded (None)
        bounds = [(0.0, None) for _ in range(M)] + [(None, None)]

        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

            if res.success:
                strategy = res.x[:M]
                value = res.x[M]
                # Normalize just in case
                strategy = strategy / np.sum(strategy)
                return GameResult(
                    optimal_strategy=strategy,
                    game_value=value,
                    is_solved=True,
                    method="simplex_lp"
                )
            else:
                logger.warning(f"Nash LP failed: {res.message}")
                return self._heuristic_solution(payoff_matrix)

        except Exception as e:
            logger.error(f"Nash Solver Error: {e}")
            return self._heuristic_solution(payoff_matrix)

    def _heuristic_solution(self, matrix: np.ndarray) -> GameResult:
        """Fallback: Assumes uniform probability or max-min if simple."""
        M, N = matrix.shape
        # Simple heuristic: Uniform distribution
        strategy = np.ones(M) / M
        # Estimated value against best counter-play
        # If we play uniform, opponent minimizes our gain by choosing min column average
        # value = min_j (Sum_i (1/M * A_ij))
        value = np.min(np.mean(matrix, axis=0))

        return GameResult(
            optimal_strategy=strategy,
            game_value=value,
            is_solved=False,
            method="uniform_fallback"
        )

    def simulate_game(self, bettor_actions: list[str], bookie_actions: list[str],
                      payoff_matrix: np.ndarray) -> dict:
        """
        Simulates the game and returns interpretable results.
        """
        res = self.solve_nash(payoff_matrix)

        strategy_dict = {}
        for i, action in enumerate(bettor_actions):
            if i < len(res.optimal_strategy):
                strategy_dict[action] = round(res.optimal_strategy[i], 4)

        return {
            "strategy": strategy_dict,
            "expected_value": round(res.game_value, 4),
            "method": res.method
        }
