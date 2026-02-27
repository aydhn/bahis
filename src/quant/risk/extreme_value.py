"""
extreme_value.py – Extreme Value Theory (EVT) for Tail Risk.

Standard deviation (Normal Distribution) underestimates the risk of black swans
(e.g., 20% drops, unexpected upsets). EVT focuses explicitly on the tails
of the distribution to model extreme events accurately.

Methods:
  - Block Maxima (GEV - Generalized Extreme Value): Models the maximum of large blocks of data.
  - Peaks Over Threshold (GPD - Generalized Pareto Distribution): Models all values exceeding a high threshold.

Usage:
    analyzer = ExtremeValueAnalyzer()
    # Fit GPD to losses exceeding threshold
    var, es = analyzer.calculate_tail_risk(losses, confidence=0.99)
"""
from __future__ import annotations

import numpy as np
from loguru import logger
from dataclasses import dataclass

try:
    from scipy.stats import genpareto, genextreme
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False
    logger.warning("scipy.stats not found – ExtremeValueAnalyzer limited.")

@dataclass
class TailRiskReport:
    var_99: float = 0.0        # Value at Risk at 99%
    es_99: float = 0.0         # Expected Shortfall at 99%
    shape_param: float = 0.0   # Shape parameter (xi) - determines tail heaviness
    method: str = "empirical"
    is_fat_tailed: bool = False

class ExtremeValueAnalyzer:
    """
    Analyzes tail risk using Extreme Value Theory.
    """

    def __init__(self):
        logger.debug("ExtremeValueAnalyzer initialized.")

    def analyze_losses(self, losses: np.ndarray, threshold_quantile: float = 0.90) -> TailRiskReport:
        """
        Analyzes the tail of the loss distribution using GPD (Peaks Over Threshold).

        Args:
            losses: Array of positive loss values (e.g., -PnL where PnL < 0).
            threshold_quantile: Quantile to set the threshold (u). E.g., 0.90 uses top 10% losses.

        Returns:
            TailRiskReport with VaR and ES.
        """
        if len(losses) < 20:
            return self._empirical_risk(losses)

        # Only consider actual losses (positive values representing magnitude of loss)
        # Assuming input 'losses' contains magnitudes of negative returns

        # Determine threshold u
        u = np.quantile(losses, threshold_quantile)

        # Excesses over threshold
        excesses = losses[losses > u] - u

        if len(excesses) < 10 or not SCIPY_OK:
            return self._empirical_risk(losses)

        try:
            # Fit GPD: genpareto.fit(data) -> (shape, loc, scale)
            # shape (xi): >0 (Frechet/Fat), 0 (Gumbel), <0 (Weibull/Thin)
            # We fix loc=0 for excesses usually
            c, loc, scale = genpareto.fit(excesses, floc=0)

            # Calculate VaR and ES at 99%
            # Formula: VaR_p = u + (scale/c) * ( ((n/Nu) * (1-p))^(-c) - 1 )
            # Where n=total samples, Nu=exceedances

            p = 0.99
            n = len(losses)
            nu = len(excesses)

            if c != 0:
                var_99 = u + (scale / c) * ( ((n / nu) * (1 - p))**(-c) - 1 )
                # ES_p = (VaR_p + scale - c*u) / (1-c) approx for GPD mean excess?
                # Formula: ES_p = (VaR_p / (1-c)) + (scale - c*u) / (1-c)
                # Simpler: ES = VaR + scale/(1-xi) * ...
                # Standard formula: ES = (VaR + scale - c*u) / (1-c)  <-- Only if VaR > u
                es_99 = (var_99 + scale - c * u) / (1 - c)
            else:
                var_99 = u - scale * np.log((n/nu)*(1-p))
                es_99 = var_99 + scale

            return TailRiskReport(
                var_99=float(var_99),
                es_99=float(es_99),
                shape_param=float(c),
                method="gpd",
                is_fat_tailed=(c > 0.1) # Arbitrary threshold for "fat"
            )

        except Exception as e:
            logger.warning(f"GPD fit failed: {e}")
            return self._empirical_risk(losses)

    def _empirical_risk(self, losses: np.ndarray) -> TailRiskReport:
        """Fallback: Historical Simulation."""
        if len(losses) == 0:
            return TailRiskReport()

        var_99 = np.percentile(losses, 99)
        # ES is mean of losses > VaR
        tail_losses = losses[losses >= var_99]
        es_99 = np.mean(tail_losses) if len(tail_losses) > 0 else var_99

        return TailRiskReport(
            var_99=float(var_99),
            es_99=float(es_99),
            method="empirical"
        )
