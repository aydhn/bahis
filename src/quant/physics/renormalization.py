"""
renormalization.py – Renormalization Group (RG) Flow for Momentum.

Applies the Renormalization Group (RG) theory from statistical physics to analyze
momentum fluctuations across different time scales.
Detects "Critical Points" (Phase Transitions) where microscopic actions (a pass, a shot)
scale up to macroscopic changes (a goal).

Concepts:
  - Coarse Graining: Averaging momentum over larger time windows (1min -> 5min -> 15min).
  - Scaling Exponent (Beta Function): How variance changes with scale.
  - Critical Point: When the system becomes scale-invariant (Self-Organized Criticality),
    a phase transition (Goal) is imminent.
"""
import numpy as np
from typing import Dict, List, Optional
from loguru import logger

class RenormalizationGroup:
    """
    RG Flow Analyzer for high-frequency match momentum.
    """

    def __init__(self, scales: List[int] = [1, 3, 5, 10]):
        self.scales = scales  # Time windows in minutes

    def analyze_flow(self, momentum_series: List[float]) -> Dict[str, Any]:
        """
        Analyzes the RG flow of the momentum time series.

        Args:
            momentum_series: List of momentum values (e.g. per minute).

        Returns:
            Dict containing critical point status and scaling exponents.
        """
        if len(momentum_series) < max(self.scales) * 2:
            return {"status": "insufficient_data"}

        data = np.array(momentum_series)
        variances = []

        # Coarse Graining Loop
        for scale in self.scales:
            # Reshape into chunks of size 'scale'
            # Truncate to multiple of scale
            n_chunks = len(data) // scale
            truncated = data[:n_chunks * scale]
            reshaped = truncated.reshape((n_chunks, scale))

            # Coarse grained variable: Block Average
            coarse = np.mean(reshaped, axis=1)

            # Calculate variance (susceptibility)
            var = np.var(coarse)
            variances.append(var)

        # Fit Power Law: Var(s) ~ s^(-beta)
        # log(Var) = -beta * log(s) + C

        log_scales = np.log(np.array(self.scales))
        log_vars = np.log(np.array(variances) + 1e-9)

        # Linear Regression
        slope, intercept = np.polyfit(log_scales, log_vars, 1)
        beta = -slope

        # Interpretation
        # beta = 1.0 -> 1/f noise (Critical State / Pink Noise)
        # beta = 0.5 -> Random Walk (uncorrelated)
        # beta > 1.5 -> Strong Persistence

        status = "stable"
        criticality = 0.0

        if 0.8 <= beta <= 1.2:
            status = "critical"
            criticality = 1.0 - abs(1.0 - beta) # 1.0 is peak
        elif beta > 1.5:
            status = "super_stable"
        else:
            status = "noise"

        return {
            "beta": round(beta, 4),
            "status": status,
            "criticality": round(criticality, 2),
            "variances": [round(v, 4) for v in variances]
        }

    def check_phase_transition(self, momentum_history: List[float], match_id: str) -> Optional[str]:
        """
        Detects if a goal (Phase Transition) is likely.
        """
        res = self.analyze_flow(momentum_history)

        if res.get("status") == "critical":
            logger.info(f"RG Flow: Critical Point detected for {match_id} (Beta={res['beta']})")
            return "GOAL_LIKELY"

        return None
