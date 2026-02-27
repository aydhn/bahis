"""
black_litterman_optimizer.py - Portfolio Optimization Engine.

Implements the Black-Litterman model to combine market-implied probabilities (Prior)
with our proprietary model predictions (Views), adjusting for Epistemic Uncertainty.
It outputs optimal stake weights for a portfolio of concurrent bets.
"""

import numpy as np
from loguru import logger
from typing import Dict, Any, List

class BlackLittermanOptimizer:
    """
    Combines market priors and model views to generate optimal allocations.
    """

    def __init__(self, tau: float = 0.05, risk_aversion: float = 2.5):
        """
        tau: Scalar indicating the uncertainty of the prior (market). Usually small (0.01 - 0.05).
        risk_aversion: Risk aversion coefficient (lambda). Higher = more conservative.
        """
        self.tau = tau
        self.risk_aversion = risk_aversion

    def calculate_single_asset_multiplier(self, implied_prob: float, model_prob: float, epistemic_uncertainty: float) -> float:
        """
        Simplified 1D Black-Litterman approach to generate a conviction multiplier for a single bet.

        Args:
            implied_prob: The probability implied by market odds (1 / decimal_odds).
            model_prob: The probability output by our predictive model.
            epistemic_uncertainty: The uncertainty of our model's prediction (0.0 to 1.0).

        Returns:
            A multiplier (float) to be applied to the base stake.
            - > 1.0: High conviction (Model is sure, and differs from market favorably).
            - < 1.0: Low conviction (Model is unsure, or agrees closely with market).
        """
        try:
            if implied_prob <= 0 or implied_prob >= 1:
                return 1.0
            if model_prob <= 0 or model_prob >= 1:
                return 1.0

            # Variance of the market (Prior). We assume a Bernoulli-like variance for simplicity in 1D.
            # var_prior = p * (1-p)
            var_prior = implied_prob * (1.0 - implied_prob)

            # Variance of our view (Omega). Scaled by epistemic uncertainty.
            # If epistemic uncertainty is high, Omega is large (we trust our view less).
            omega = (epistemic_uncertainty + 0.01) * var_prior # Base scaling + epsilon

            # The BL formula in 1D for the blended expected return (posterior mean):
            # mu_bl = [ (tau * var_prior)^-1 + omega^-1 ]^-1 * [ (tau * var_prior)^-1 * implied_prob + omega^-1 * model_prob ]

            p_tau_var = 1.0 / (self.tau * var_prior)
            p_omega = 1.0 / omega

            blended_prob = (p_tau_var * implied_prob + p_omega * model_prob) / (p_tau_var + p_omega)

            # Calculate the "Edge" based on the blended probability
            blended_edge = (blended_prob / implied_prob) - 1.0

            # If the blended edge is negative, we shouldn't bet, multiplier is 0
            if blended_edge <= 0:
                return 0.0

            # Base multiplier logic:
            # We want to scale relative to a "standard" edge, say 5%
            standard_edge = 0.05
            multiplier = blended_edge / standard_edge

            # Apply risk aversion dampening
            multiplier = multiplier / self.risk_aversion

            # Clip the multiplier to sane bounds (e.g., 0 to 2.0x)
            return float(np.clip(multiplier, 0.0, 2.0))

        except Exception as e:
            logger.error(f"Black-Litterman 1D calculation failed: {e}")
            return 1.0

    def allocate_portfolio(self, matches: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Full multi-asset Black-Litterman allocation.
        Not fully implemented yet, placeholder for future multi-match concurrent optimization.
        """
        # Placeholder for future full matrix math implementation
        weights = {}
        for match in matches:
             match_id = match.get("match_id", "Unknown")
             weights[match_id] = 1.0 / len(matches) if matches else 0.0
        return weights
