"""
kelly_benter_optimizer.py - Kelly-Benter Hybrid Portfolio Optimizer.

Combines Bill Benter's confidence principles with strict fund-manager risk guardrails
to calculate precise capital allocations.
"""
from typing import Dict, Any, List
from loguru import logger
from src.quant.risk.kelly import AdaptiveKelly


class KellyBenterOptimizer:
    """
    Combines Bill Benter's edge-based staking principles with modern fund manager guardrails.
    Enhanced with Aggressive JP Morgan Fund Manager logic.
    """
    def __init__(self, base_fraction: float = 0.20, window_size: int = 50):
        self.adaptive_kelly = AdaptiveKelly(base_fraction=base_fraction, window_size=window_size)
        self.base_max_exposure = 0.15 # Max portfolio exposure
        self.dynamic_max_exposure = self.base_max_exposure

    def update_outcome(self, data: Dict[str, Any]):
        self.adaptive_kelly.update_outcome(data)

    def update_market_regime(self, volatility: float, drawdown: float):
        """
        Dynamically adjusts maximum portfolio exposure based on regime.
        """
        if drawdown > 0.10:
             # Deep drawdown: bunker mode
             self.dynamic_max_exposure = self.base_max_exposure * 0.5
        elif volatility > 0.05:
             # High volatility: reduce risk
             self.dynamic_max_exposure = self.base_max_exposure * 0.75
        elif drawdown < 0.02 and volatility < 0.02:
             # Favorable conditions: aggressive mode
             self.dynamic_max_exposure = self.base_max_exposure * 1.5
        else:
             self.dynamic_max_exposure = self.base_max_exposure

        logger.debug(f"KellyBenterOptimizer: Dynamic max exposure set to {self.dynamic_max_exposure:.2%}")

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


        # Benter's adjustment: Scale by confidence and applying strict guardrails
        benter_f = self.adaptive_kelly.calculate_fraction(p, b, 1.0) * confidence

        # Apply a non-linear confidence boost for high conviction (Aggressive Yield)
        if confidence > 0.8 and edge > 0.05:
             benter_f *= 1.25 # 25% boost on absolute best setups

        return min(benter_f, self.dynamic_max_exposure)

    def optimize_portfolio(self, opportunities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Optimizes a portfolio of opportunities using the Benter-Kelly approach.
        """
        if not opportunities:
            return []

        total_kelly = 0.0
        optimized = []

        # Sort by edge to prioritize best opportunities when scaling down
        sorted_opps = sorted(
            opportunities,
            key=lambda x: (x.get("prob", 0.0) * x.get("odds", 0.0) - 1.0) * x.get("confidence", 0.5),
            reverse=True
        )

        for opp in sorted_opps:
            p = opp.get("prob", 0.0)
            b = opp.get("odds", 0.0)
            conf = opp.get("confidence", 0.5)

            k_cap = self.calculate_fraction(p, b, confidence=conf)

            # If adding this pushes us way over, we scale it down heavily
            if total_kelly + k_cap > self.dynamic_max_exposure * 1.2:
                 k_cap *= 0.1 # Severely penalize late additions to prevent overexposure

            total_kelly += k_cap

            opp_copy = opp.copy()
            opp_copy["kelly_cap"] = k_cap
            optimized.append(opp_copy)

        # Global risk constraint strict enforcement
        if total_kelly > self.dynamic_max_exposure:
            scale_factor = self.dynamic_max_exposure / total_kelly
            logger.info(f"Kelly-Benter: Total exposure {total_kelly:.2%} exceeds dynamic max {self.dynamic_max_exposure:.2%}. Scaling by {scale_factor:.2f}")
            for opp in optimized:
                opp["kelly_cap"] *= scale_factor
                if "stake_amount" in opp:
                    opp["stake_amount"] *= scale_factor

        return optimized
