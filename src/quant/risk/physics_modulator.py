"""
physics_modulator.py – Centralized Physics-Based Risk Modulation.

This module encapsulates the complex logic for adjusting betting stakes based on
advanced physics engines (Chaos, Topology, Fractal, Ricci, etc.).
It acts as a facade for the RiskStage, keeping the pipeline clean.
"""
from typing import Dict, Any, List, Optional
from loguru import logger

class PhysicsRiskModulator:
    """
    Modulates risk (stake size) based on multi-dimensional physics metrics.
    """

    def __init__(self):
        pass

    def modulate(self,
                 bet_candidate: Dict[str, Any],
                 physics_context: Dict[str, Any],
                 ricci_report: Optional[Any] = None) -> float:
        """
        Calculates a stake multiplier based on physics indicators.

        Args:
            bet_candidate: The bet being evaluated (e.g., {'match_id': '...', 'confidence': 0.8})
            physics_context: Dictionary containing physics reports for the specific match.
                             Expected keys: 'chaos_regime', 'fractal_dim', 'roughness',
                             'homology_org_diff', 'topology_cluster', 'topology_anomaly'
            ricci_report: Global systemic risk report (optional).

        Returns:
            float: Stake multiplier (0.0 to 1.0+). 0.0 means KILL signal.
        """
        match_id = bet_candidate.get("match_id", "unknown")
        multiplier = 1.0
        reasons = []

        # 1. Global Systemic Risk (Ricci Flow)
        if ricci_report:
            if ricci_report.kill_betting:
                logger.critical(f"PhysicsModulator: Ricci Kill Switch for {match_id}. Systemic Risk: {ricci_report.systemic_risk:.2f}")
                return 0.0

            if ricci_report.stress_level in ["high", "critical"]:
                risk_factor = min(max(ricci_report.systemic_risk, 0.0), 0.9)
                ricci_mult = 1.0 - risk_factor
                multiplier *= ricci_mult
                reasons.append(f"Ricci Stress ({ricci_report.stress_level}, x{ricci_mult:.2f})")

        # 2. Chaos Theory (Lyapunov Exponents)
        chaos_regime = physics_context.get("chaos_regime", "unknown")
        if chaos_regime == "chaotic":
            logger.warning(f"PhysicsModulator: Chaos Kill for {match_id}")
            return 0.0
        elif chaos_regime == "edge_of_chaos":
            multiplier *= 0.5
            reasons.append("Edge of Chaos (x0.5)")

        # 3. Topology (Anomaly Detection)
        if physics_context.get("topology_anomaly", False):
            logger.warning(f"PhysicsModulator: Topology Kill for {match_id} (Anomaly)")
            return 0.0

        # 4. Fractal Analysis (Hurst Exponent)
        # Assuming fractal_mult is passed or derived.
        # If 'fractal_regime' is random, we reduce.
        fractal_regime = physics_context.get("fractal_regime", "unknown")
        if fractal_regime == "random":
            multiplier *= 0.8
            reasons.append("Fractal Random Walk (x0.8)")

        # If a specific multiplier was pre-calculated (legacy support)
        if "fractal_mult" in physics_context:
            multiplier *= physics_context["fractal_mult"]

        # 5. Path Signature (Roughness)
        roughness = physics_context.get("roughness", 0.0)
        if roughness > 0.1:
            # High roughness -> Higher volatility/uncertainty
            rough_mult = max(0.5, 1.0 - (roughness * 2))
            multiplier *= rough_mult
            reasons.append(f"Path Roughness ({roughness:.3f}, x{rough_mult:.2f})")

        # 6. Homology & GCN (Coordination)
        # Using pre-calculated coordination bonus/penalty if available, or raw metrics
        # Context expects 'homology_org_diff' or specific flags
        home_org = physics_context.get("home_org", 0.0)
        away_panicking = physics_context.get("away_panicking", False)

        # Heuristic: If betting HOME and Home is organized & Away panicking -> Boost
        # Note: bet_candidate might not have 'selection' easily available if it's raw ensemble result.
        # Assuming we are evaluating the primary prediction direction.
        # For simplicity, we apply a general coordination adjustment if distinct.

        org_diff = physics_context.get("homology_org_diff", 0.0)
        if abs(org_diff) > 0.2:
            # If significant difference in organization, maybe slight boost or penalty?
            # Keeping it conservative: we don't boost stake above 1.0 purely on physics often,
            # but we penalize if our side is disorganized.
            pass

        # Log modulation
        if abs(multiplier - 1.0) > 0.01:
            logger.info(f"PhysicsModulator {match_id}: Stake x{multiplier:.2f} | Reasons: {', '.join(reasons)}")

        return max(0.0, multiplier)
