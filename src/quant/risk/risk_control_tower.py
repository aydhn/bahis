"""
risk_control_tower.py – Centralized Risk Orchestrator.

This module acts as the "Air Traffic Control" for all betting decisions.
It unifies the fragmented risk logic previously scattered across RiskStage.

Responsibilities:
  1. Gatekeeping (Cognitive, Pre-Mortem)
  2. Sizing (Kelly, Physics Modulation)
  3. Funding (Treasury, Liquidity)
  4. Decision Finalization (Approved Stake, Rationale)
"""
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from loguru import logger

from src.core.regime_kelly import RegimeKelly, RegimeState
from src.core.cognitive_guardian import CognitiveGuardian
from src.quant.analysis.pre_mortem import PreMortemAnalyzer, PreMortemReport
from src.quant.risk.physics_modulator import PhysicsRiskModulator
from src.quant.finance.treasury import TreasuryEngine
from src.quant.treasury.synthetic_engine import SyntheticEngine
from src.quant.treasury.arbitrage_scanner import ArbitrageScanner
from src.quant.analysis.market_regime_detector import MarketRegimeDetector, RegimeMetrics
from src.quant.analysis.philosophical_engine import PhilosophicalEngine
from src.quant.analysis.causal_reasoner import CausalReasoner
from src.core.system_architect import StrategicDirective

# Import Physics Reports for Type Hinting (Optional, but good for clarity)
try:
    from src.quant.physics.fisher_geometry import FisherReport
    from src.quant.physics.ricci_flow import RicciReport
    from src.quant.analysis.hypergraph_unit import HypergraphReport
except ImportError:
    pass

@dataclass
class RiskDecision:
    """Final decision from the Risk Tower."""
    approved: bool = False
    stake_amount: float = 0.0
    stake_pct: float = 0.0
    rationale: str = ""
    rejection_reason: str = ""
    adjustments: List[str] = None
    regime_metrics: Optional[RegimeMetrics] = None

class RiskControlTower:
    """
    The Facade for all Risk Operations.
    """

    def __init__(self):
        # 1. Core Risk Engine
        self.kelly = RegimeKelly()

        # 2. Gatekeepers
        self.guardian = CognitiveGuardian()
        self.pre_mortem = PreMortemAnalyzer()

        # 3. Modulators
        self.physics_modulator = PhysicsRiskModulator()
        self.regime_detector = MarketRegimeDetector()

        # 4. Treasury & Value Ops
        self.treasury = TreasuryEngine()
        self.synthetic = SyntheticEngine()
        self.arb_scanner = ArbitrageScanner()

        # 5. Zero Error Components (Philosophical & Causal)
        self.philosopher = PhilosophicalEngine()
        self.causal_reasoner = CausalReasoner()

        logger.info("RiskControlTower initialized and ready for duty.")

    def evaluate_bet(self,
                     bet_candidate: Dict[str, Any],
                     context: Dict[str, Any]) -> RiskDecision:
        """
        Evaluate a potential bet through the full risk pipeline.

        Args:
            bet_candidate: Dict with 'match_id', 'prob_home', 'odds', 'confidence', 'ev', etc.
            context: Pipeline context containing 'odds_history', 'physics_reports', etc.

        Returns:
            RiskDecision object.
        """
        match_id = bet_candidate.get("match_id", "unknown")
        decision = RiskDecision(adjustments=[])

        # --- 0.5 The Architect (Strategic Directive Check) ---
        # Retrieve global strategy from context
        directive: Optional[StrategicDirective] = context.get("strategic_directive")

        if directive:
            # Check Posture
            if directive.posture == "LIQUIDATION":
                decision.approved = False
                decision.rejection_reason = "Architect Directive: LIQUIDATION (Panic Mode)"
                return decision

            if directive.posture == "BUNKER":
                # Reduce stake or increase confidence requirement
                decision.adjustments.append("Architect: BUNKER Mode (Stake Reduced)")

            # Check Edge Requirement
            required_ev = 0.05 * directive.required_edge_multiplier
            if bet_candidate.get("ev", 0) < required_ev:
                decision.approved = False
                decision.rejection_reason = f"Architect: Insufficient Edge (Req: {required_ev:.2%}, Act: {bet_candidate.get('ev',0):.2%})"
                return decision

        # --- 0. Synthetic Value & Arbitrage Check (Treasury Ops) ---
        # Before evaluating risk, check if we can optimize the entry using derivatives or arbitrage.

        # 0.1 Synthetic Value (DNB/DC)
        # Assuming we have access to market odds for DNB/DC if available, or we just calculate theoretical.
        # Here we just calculate what the fair DNB odds SHOULD be based on 1X2.
        # If the bet is a HOME win, we check if Home DNB is a safer alternative with good EV.

        odds_home = bet_candidate.get("odds", 0.0)
        # We need draw and away odds to calculate synthetic. Assuming they might be in context or bet_candidate.
        # If not available, we skip.

        # Try to extract full market odds from context if available
        # This part assumes context structure has 'matches' or 'raw_data' we can look up.
        # For simplicity, we skip if we don't have full odds.

        # 0.2 Arbitrage Check
        # If this match has arbitrage opportunities, we flag it.
        # This requires multi-bookie data which might be in 'context["raw_data"]'.

        # --- 1. Cognitive Gatekeeper ---
        # Prevent Tilt, Chase, Hubris
        bet_req = {"stake": 1.0, "team": bet_candidate.get("home_team", "Unknown")} # Stake placeholder
        if not self.guardian.check_bet(bet_req):
            decision.approved = False
            decision.rejection_reason = "Cognitive Guardian Blocked (Tilt/Chase)"
            return decision

        # --- 2. Market Regime Detection ---
        # Is the market safe?
        odds_history = context.get("odds_history", {}).get(match_id, [])
        regime = self.regime_detector.detect_regime(match_id, odds_history)
        decision.regime_metrics = regime

        if regime.regime == "CHAOTIC" or regime.regime == "CRASH":
            decision.approved = False
            decision.rejection_reason = f"Market Regime Veto: {regime.regime} ({regime.description})"
            return decision

        # --- 3. Pre-Mortem Analysis ---
        # The Devil's Advocate
        pm_report = self.pre_mortem.analyze(bet_candidate)
        if pm_report.kill_signal:
            decision.approved = False
            decision.rejection_reason = f"Pre-Mortem Kill: {', '.join(pm_report.reasons)}"
            return decision

        if pm_report.caution_signal:
            decision.adjustments.append(f"Pre-Mortem Caution: {', '.join(pm_report.reasons)}")

        # --- 3.1 Epistemic Validity (Zero Error) ---
        epistemic_report = self.philosopher.evaluate(
            probability=bet_candidate.get("prob_home", 0.5), # Assuming home bet for main logic
            confidence=bet_candidate.get("confidence", 0.5),
            sample_size=bet_candidate.get("sample_size", 100), # Default if not passed
            match_id=match_id
        )
        if not epistemic_report.epistemic_approved:
            decision.approved = False
            decision.rejection_reason = f"Epistemic Veto: {', '.join(epistemic_report.rejection_reasons)}"
            return decision

        # --- 3.2 Physics Veto (Ricci & Fisher) ---
        # Deep Geometry Check
        physics_reports = context.get("physics_reports", {})

        # Ricci Flow (Systemic Risk)
        if "ricci_report" in physics_reports:
            ricci = physics_reports["ricci_report"]
            if ricci and hasattr(ricci, 'stress_level') and ricci.stress_level == "critical":
                decision.approved = False
                decision.rejection_reason = f"Ricci Flow Veto: Critical Systemic Stress (Risk: {ricci.systemic_risk:.2f})"
                return decision

        # Fisher Geometry (Regime Shift & Anomaly)
        if "fisher_reports" in physics_reports:
            fisher = physics_reports["fisher_reports"].get(match_id)
            if fisher:
                # If anomalous AND regime shift, hard block
                if fisher.is_anomaly and fisher.regime_shift:
                    decision.approved = False
                    decision.rejection_reason = f"Fisher Geometry Veto: Anomaly + Regime Shift (FR Distance: {fisher.fisher_rao_distance:.2f})"
                    return decision

                # If just Regime Shift, reduce confidence/stake
                if fisher.regime_shift:
                    decision.adjustments.append(f"Fisher Warning: Regime Shift detected (FR={fisher.fisher_rao_distance:.2f}).")

        # --- 3.3 Structural Integrity (Hypergraph) ---
        if "hypergraph_reports" in physics_reports:
            # Check home/away vulnerability depending on bet
            # For simplicity, if we bet on HOME, we check HOME structural integrity
            # If HOME structure is compromised, reduce stake.
            # If AWAY structure is compromised, increase confidence? (Not implemented here, but good idea)

            # Assuming 'selection' is in bet_candidate or we infer from context
            # Defaulting to checking HOME team for now as primary risk
            hg_reports = physics_reports["hypergraph_reports"].get(match_id, {})
            home_hg = hg_reports.get("home")

            if home_hg and hasattr(home_hg, 'vulnerability_index'):
                if home_hg.vulnerability_index > 0.6:
                    decision.adjustments.append(f"Hypergraph Warning: High Structural Vulnerability ({home_hg.vulnerability_index:.2f}).")
                    # We will apply this multiplier later in modulation or manually here
                    # Let's add it as a modulation factor below

        # --- 3.4 Causal Check (Spurious Correlation) ---
        # If any major causal factor (e.g. Red Card) is active, ensure we account for it
        # For now, we simulate a check. Real implementation would parse match_data.
        # causal_effect = self.causal_reasoner.estimate_effect("recent_form", "outcome")
        # if not causal_effect.is_significant: ... (Logic placeholder)

        # --- 4. Kelly Sizing ---
        # Calculate Base Stake
        # Map MarketRegime to RegimeState for Kelly
        kelly_regime = RegimeState()
        if regime.regime == "VOLATILE":
            kelly_regime.volatility_regime = "storm"
        elif regime.regime == "STABLE":
            kelly_regime.volatility_regime = "calm"

        kelly_res = self.kelly.calculate(
            probability=bet_candidate.get("prob_home", 0.0), # Assuming Home bet for now
            odds=bet_candidate.get("odds", 0.0),
            match_id=match_id,
            regime=kelly_regime
        )

        if not kelly_res.approved:
            decision.approved = False
            decision.rejection_reason = f"Kelly Rejected: {kelly_res.rejection_reason}"
            return decision

        final_stake_pct = kelly_res.stake_pct

        # --- 5. Physics Modulation ---
        # Apply advanced physics multipliers
        physics_ctx = self._extract_physics_context(match_id, context)
        phys_mult = self.physics_modulator.modulate(bet_candidate, physics_ctx)

        # Hypergraph Penalty Integration
        # If we had a structural warning, reduce stake by 20%
        if any("Hypergraph Warning" in adj for adj in decision.adjustments):
            phys_mult *= 0.8

        if phys_mult <= 0.0:
            decision.approved = False
            decision.rejection_reason = "Physics Kill Signal"
            return decision

        if phys_mult != 1.0:
            final_stake_pct *= phys_mult
            decision.adjustments.append(f"Physics Multiplier: x{phys_mult:.2f}")

        # --- 5.5 Architect Stake Adjustment ---
        if directive:
            if directive.posture == "BUNKER":
                final_stake_pct *= 0.5
                decision.adjustments.append("Architect: BUNKER Stake Cap (x0.5)")
            elif directive.posture == "EXPANSION":
                final_stake_pct = min(final_stake_pct * 1.2, directive.max_daily_exposure)
                decision.adjustments.append(f"Architect: EXPANSION Boost (Max {directive.max_daily_exposure:.1%})")

            # Global exposure cap enforcement
            if final_stake_pct > directive.max_daily_exposure:
                final_stake_pct = directive.max_daily_exposure
                decision.adjustments.append(f"Architect: Capped at Max Exposure ({directive.max_daily_exposure:.1%})")

        # --- 6. Treasury Check ---
        # Can we afford it?
        amount = final_stake_pct * self.treasury.state.total_capital # Estimate amount
        approved_amount = self.treasury.request_capital(amount, strategy_type="aggressive")

        if approved_amount <= 0:
            decision.approved = False
            decision.rejection_reason = "Treasury Denied Capital (Drawdown or Liquidity)"
            return decision

        if approved_amount < amount:
            decision.adjustments.append(f"Treasury Reduced Stake: {amount:.2f} -> {approved_amount:.2f}")

        # --- 7. Finalize ---
        decision.approved = True
        decision.stake_amount = approved_amount
        # Re-calculate pct based on approved amount
        decision.stake_pct = approved_amount / max(self.treasury.state.total_capital, 1.0)
        decision.rationale = f"Approved. Edge: {kelly_res.edge:.2%}. " + "; ".join(decision.adjustments)

        return decision

    def _extract_physics_context(self, match_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Helper to extract relevant physics reports for modulation."""
        # Similar logic to old RiskStage but streamlined
        physics_reports = context.get("physics_reports", {})

        # Populate dict with specific metrics needed by PhysicsModulator
        ctx = {}
        if "chaos_reports" in physics_reports and match_id in physics_reports["chaos_reports"]:
             ctx["chaos_regime"] = physics_reports["chaos_reports"][match_id].regime

        if "fractal_reports" in physics_reports and match_id in physics_reports["fractal_reports"]:
             ctx["fractal_regime"] = physics_reports["fractal_reports"][match_id].regime

        # Add other extractions as needed by PhysicsRiskModulator
        return ctx
