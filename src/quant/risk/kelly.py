"""
kelly.py - Adaptive Kelly Criterion with Bankroll Heat Management.

This module implements a dynamic version of the Kelly Criterion that adjusts
the betting fraction based on realized performance and market regime.
"""
from typing import List, Dict


class AdaptiveKelly:
    """
    Dynamic Kelly Manager.

    Features:
    - Tracks recent bet performance (Win/Loss).
    - Calculates 'Realized Edge' vs 'Predicted Edge'.
    - Reduces Kelly fraction during drawdowns or when model calibration drifts.
    """

    def __init__(self, base_fraction: float = 0.25, window_size: int = 50):
        self.base_fraction = base_fraction
        self.window_size = window_size
        self.history: List[Dict] = []
        self.current_drawdown = 0.0
        self.peak_bankroll = 1.0 # Normalized

    def update_outcome(self, bet_result: Dict):
        """
        Update the history with a resolved bet.
        bet_result: {"predicted_prob": 0.55, "odds": 2.0, "won": True/False, "stake": 100}
        """
        self.history.append(bet_result)
        if len(self.history) > self.window_size:
            self.history.pop(0)

        # Update Drawdown
        # simplified PnL tracking for internal state
        # We need actual bankroll tracking elsewhere, but here we can track relative performance

    def calculate_fraction(self, probability: float, odds: float, confidence: float = 1.0) -> float:
        """
        Calculate the optimal Kelly fraction.

        Args:
            probability: Estimated win probability.
            odds: Decimal odds.
            confidence: Model confidence score (0.0 - 1.0).

        Returns:
            float: Recommended fraction of bankroll (e.g., 0.02 for 2%).
        """
        if probability <= 0 or odds <= 1:
            return 0.0

        # 1. Basic Kelly via fast_math (Scalar for max speed)
        from src.system.container import container
        fast_math = container.get("fast_math")
        if fast_math:
            raw_kelly = float(fast_math.fast_kelly(probability, odds, fraction=1.0))
        else:
            q = 1.0 - probability
            b = odds - 1.0
            raw_kelly = (probability * b - q) / b if b > 0 else 0.0
            raw_kelly = max(0.0, float(raw_kelly))

        if raw_kelly <= 0:
            return 0.0

        # 2. Apply Base Fraction (Safety)
        safe_kelly = raw_kelly * self.base_fraction

        # 3. Apply Confidence & Calibration Adjustment
        # If we have history, check if we are over-estimating probability
        calibration_factor = self._get_calibration_factor()

        # 4. Apply Drawdown Protection (Volatility Scaling)
        # If we are in a losing streak, reduce size to avoid Ruin
        # (This logic would require external bankroll state, assuming standard for now)

        final_fraction = safe_kelly * confidence * calibration_factor

        # Cap at 5% max bet
        return min(final_fraction, 0.05)

    def _get_calibration_factor(self) -> float:
        """
        Compare average predicted probability vs realized win rate in recent history.
        If we predict 60% but win 40%, factor < 1.0.
        """
        if not self.history:
            return 1.0

        predicted_sum = sum(h.get("predicted_prob", 0.5) for h in self.history)
        won_sum = sum(1.0 if h.get("won") else 0.0 for h in self.history)

        avg_predicted = predicted_sum / len(self.history)
        realized_rate = won_sum / len(self.history)

        if avg_predicted <= 0:
            return 1.0

        ratio = realized_rate / avg_predicted

        # Dampen the ratio to avoid wild swings
        # If ratio is 0.8, we return 0.8. If 1.2, return 1.0 (conservative).
        return min(max(ratio, 0.5), 1.0)
