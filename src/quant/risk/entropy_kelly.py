import numpy as np
from src.quant.risk.kelly import AdaptiveKelly
from src.system.container import container

class EntropyKelly(AdaptiveKelly):
    """
    Entropy-Adjusted Kelly Manager.
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
        valid_probs = [p for p in probs if p > 0.0]
        if not valid_probs:
            return 1.0 # Max uncertainty

        total = sum(valid_probs)
        normalized_probs = np.array([p/total for p in valid_probs], dtype=np.float64)

        fast_math = container.get("fast_math")
        if fast_math:
            entropy = fast_math.fast_entropy(normalized_probs)
        else:
            # Fallback
            entropy = -np.sum(normalized_probs * np.log2(normalized_probs))

        # Max entropy for N outcomes is log2(N)
        max_entropy = np.log2(len(probs))

        if max_entropy == 0:
            return 0.0

        return entropy / max_entropy

    def calculate_fraction(self, probability: float, odds: float, confidence: float = 1.0,
                           all_probs: list[float] = None) -> float:
        """
        Calculate Kelly fraction with Entropy Penalty.
        """
        # 1. Get Base Adaptive Kelly
        base_kelly = super().calculate_fraction(probability, odds, confidence)

        if base_kelly <= 0:
            return 0.0

        # 2. Calculate Entropy Scalar
        entropy_scalar = 1.0
        if all_probs and len(all_probs) > 1:
            h_norm = self.calculate_entropy(all_probs)
            entropy_scalar = 1.0 - (0.5 * h_norm)

        final_stake = base_kelly * entropy_scalar

        return final_stake
