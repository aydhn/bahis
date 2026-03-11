"""
entropy_kelly.py – Entropy-Adjusted Adaptive Kelly Criterion.

This module extends the standard Adaptive Kelly by incorporating Shannon Entropy
as a measure of market uncertainty. High entropy (e.g., 33-33-33 split) indicates
maximum uncertainty, where Kelly sizing should be reduced even if an edge exists.
"""
import numpy as np
from src.quant.risk.kelly import AdaptiveKelly

class EntropyKelly(AdaptiveKelly):
    """
    Entropy-Adjusted Kelly Manager.

    Logic:
    - Calculate Shannon Entropy of the outcome probabilities (Home, Draw, Away).
    - Normalize Entropy (0.0 to 1.0).
    - Apply a scalar penalty based on entropy.
    - Low Entropy (high confidence in one outcome) -> Full Kelly.
    - High Entropy (coin flip) -> Reduced Kelly.
    """

    def __init__(self, base_fraction: float = 0.25, window_size: int = 50, entropy_penalty_power: float = 1.0):
        super().__init__(base_fraction, window_size)
        self.penalty_power = entropy_penalty_power

    def calculate_entropy(self, probs: list[float]) -> float:
        """
        Calculates normalized Shannon Entropy for a set of probabilities using fast_math.
        H(X) = - sum(p * log2(p))
        Normalized = H(X) / log2(N)
        """
        from src.extensions.fast_math import fast_entropy

        valid_probs = [p for p in probs if p > 0.0]
        if not valid_probs:
            return 1.0 # Max uncertainty

        total = sum(valid_probs)
        normalized_probs = np.array([p/total for p in valid_probs], dtype=np.float64)

        entropy = fast_entropy(normalized_probs)

        # Max entropy for N outcomes is log2(N)
        max_entropy = np.log2(len(probs))

        if max_entropy == 0:
            return 0.0

        return entropy / max_entropy

    def calculate_fraction(self, probability: float, odds: float, confidence: float = 1.0,
                           all_probs: list[float] = None) -> float:
        """
        Calculate Kelly fraction with Entropy Penalty.

        Args:
            probability: Prob of the selection (e.g., Home Win).
            odds: Market odds.
            confidence: Model specific confidence.
            all_probs: List of probabilities for all outcomes [p_home, p_draw, p_away].
                       Required for entropy calculation.
        """
        # 1. Get Base Adaptive Kelly
        base_kelly = super().calculate_fraction(probability, odds, confidence)

        if base_kelly <= 0:
            return 0.0

        # 2. Calculate Entropy Scalar
        entropy_scalar = 1.0
        if all_probs and len(all_probs) > 1:
            h_norm = self.calculate_entropy(all_probs)

            # Use a linear penalty: Map H[0, 1] to Scalar[1.0, 0.5]
            # This ensures we don't completely kill bets on favorites (which still have some entropy),
            # but we reduce exposure on highly uncertain (flat distribution) matches.
            entropy_scalar = 1.0 - (0.5 * h_norm)

        final_stake = base_kelly * entropy_scalar

        return final_stake
