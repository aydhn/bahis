"""
kelly_benter_optimizer.py - Kelly-Benter Hybrid Portfolio Optimizer.

Combines Bill Benter's confidence principles with strict fund-manager risk guardrails
to calculate precise capital allocations.
"""
from typing import Dict, Any, List
import numpy as np
from loguru import logger
from scipy.optimize import minimize
from src.quant.risk.kelly import AdaptiveKelly

class KellyBenterOptimizer:
    def update_outcome(self, data: Dict[str, Any]):
        self.adaptive_kelly.update_outcome(data)

    """
    Combines Bill Benter's edge-based staking principles with modern fund manager guardrails.
    """
    def __init__(self, base_fraction: float = 0.20, window_size: int = 50):
        self.adaptive_kelly = AdaptiveKelly(base_fraction=base_fraction, window_size=window_size)
        self.max_exposure = 0.15 # Max portfolio exposure

    def calculate_fraction(self, p: float, b: float, confidence: float = 0.5) -> float:
        """
        Calculates the Benter-adjusted Kelly fraction.
        Benter approach heavily relies on edge and probability confidence.
        """
        if b <= 1.0 or p <= 0.0 or p >= 1.0:
            return 0.0

        edge = (p * b) - 1.0
        if edge <= 0.0:
            return 0.0

        # Base Kelly fraction
        base_f = edge / (b - 1.0)

        # Benter's adjustment: Scale by confidence and applying strict guardrails
        # High confidence -> take more of the Kelly fraction
        # Low confidence -> take significantly less
        benter_f = self.adaptive_kelly.calculate_fraction(p, b, confidence) * confidence

        return min(benter_f, self.max_exposure)

    def optimize_portfolio(self, opportunities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Optimizes a portfolio of opportunities using the Benter-Kelly approach.
        """
        if not opportunities:
            return []

        # Placeholder for full Markowitz integration if needed,
        # but Kelly-Benter usually treats events independently with global caps.

        total_kelly = 0.0
        optimized = []

        for opp in opportunities:
            p = opp.get("prob", 0.0)
            b = opp.get("odds", 0.0)
            conf = opp.get("confidence", 0.5)

            k_cap = self.calculate_fraction(p, b, confidence=conf)
            total_kelly += k_cap

            opp_copy = opp.copy()
            opp_copy["kelly_cap"] = k_cap
            optimized.append(opp_copy)

        # Global risk constraint
        if total_kelly > self.max_exposure:
            scale_factor = self.max_exposure / total_kelly
            logger.info(f"Kelly-Benter: Total exposure {total_kelly:.2%} exceeds max {self.max_exposure:.2%}. Scaling by {scale_factor:.2f}")
            for opp in optimized:
                opp["kelly_cap"] *= scale_factor
                if "stake_amount" in opp:
                    opp["stake_amount"] *= scale_factor

        return optimized
