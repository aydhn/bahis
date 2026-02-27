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
from src.quant.analysis.game_theory_engine import GameTheoryEngine

# NEW: Advanced Risk Modules
from src.quant.finance.liquidity_engine import LiquidityEngine
from src.quant.finance.optimal_execution import OptimalExecutionModel
from src.quant.risk.extreme_value import ExtremeValueAnalyzer
from src.quant.finance.stress_tester import PortfolioStressTester
from src.quant.finance.black_litterman_optimizer import BlackLittermanOptimizer
from src.core.systemic_risk_covar import SystemicRiskCoVaR
import numpy as np

# NEW: Smart Money
from src.extensions.smart_money import SmartMoneyDetector

# NEW: Boardroom
from src.core.boardroom import Boardroom

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
        self.liquidity = LiquidityEngine() # NEW
        self.optimal_execution = OptimalExecutionModel() # NEW: Almgren-Chriss Slicing
        self.stress_tester = PortfolioStressTester() # NEW
        self.systemic_risk = SystemicRiskCoVaR() # NEW: Systemic Risk (CoVaR)
        self.black_litterman = BlackLittermanOptimizer() # NEW: Portfolio Optimizer

        # 5. Zero Error Components (Philosophical, Causal & EVT)
        self.philosopher = PhilosophicalEngine()
        self.causal_reasoner = CausalReasoner()
        self.evt_analyzer = ExtremeValueAnalyzer() # NEW
        self.game_theory = GameTheoryEngine() # NEW
        self.smart_money = SmartMoneyDetector() # NEW
        self.boardroom = Boardroom() # NEW

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

        # --- 0.4 Teleology Check (Biscuit & Motivation) ---
        is_biscuit = bet_candidate.get("is_biscuit", False)
        # If Biscuit Game (Mutually beneficial draw) is detected:
        # We REJECT standard Home/Away bets.
        # (Unless we specifically support Draw bets here, but RiskStage usually passes Home candidates)
        if is_biscuit:
            decision.approved = False
            decision.rejection_reason = "Teleology Veto: Biscuit Game (Mutually Beneficial Draw)."
            return decision

        # Motivation Mismatch Check
        # If mismatch > 0.6, ensure we are not betting AGAINST the motivated team.
        mismatch = bet_candidate.get("motivation_mismatch", 0.0)
        if mismatch > 0.6:
            h_mot = bet_candidate.get("home_motivation", 5.0)
            a_mot = bet_candidate.get("away_motivation", 5.0)

            # Check bet selection to apply motivation logic correctly
            # Default to HOME if not specified (legacy behavior), but try to infer.
            selection = bet_candidate.get("selection", "HOME").upper()

            if selection == "HOME":
                if h_mot < a_mot:
                    decision.approved = False
                    decision.rejection_reason = f"Teleology Veto: Motivation Mismatch (Home {h_mot} vs Away {a_mot})."
                    return decision
                else:
                    decision.adjustments.append(f"Teleology Boost: High Motivation ({h_mot} vs {a_mot})")
            elif selection == "AWAY":
                if a_mot < h_mot:
                    decision.approved = False
                    decision.rejection_reason = f"Teleology Veto: Motivation Mismatch (Away {a_mot} vs Home {h_mot})."
                    return decision
                else:
                    decision.adjustments.append(f"Teleology Boost: High Motivation ({a_mot} vs {h_mot})")

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

        # --- 0.6 EVT Tail Risk Check (NEW) ---
        # If we have historical PnL/Loss data for this league/market, check tail risk.
        # Assuming context might have 'league_losses' or similar.
        league_losses = context.get("league_losses", []) # Placeholder
        if league_losses:
            tail_report = self.evt_analyzer.analyze_losses(league_losses)
            # If tail is extremely fat (shape param > 0.3) or VaR is huge, block or reduce.
            if tail_report.is_fat_tailed and tail_report.shape_param > 0.3:
                decision.adjustments.append(f"EVT Warning: Fat Tail (xi={tail_report.shape_param:.2f}). Stake halved.")
                # We apply this reduction later or return early if critical
                # Let's just flag it for multiplier reduction.

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
            # Strict Veto: Chaos Kill Switch
            if regime.regime == "CHAOTIC" and regime.confidence > 0.8: # High confidence chaos
                 decision.approved = False
                 decision.rejection_reason = f"Chaos Veto: Market is Chaotic (Conf: {regime.confidence:.2f})"
                 return decision

            if regime.regime == "CRASH":
                 decision.approved = False
                 decision.rejection_reason = f"Market Regime Veto: {regime.regime} (Crash Protocol)"
                 return decision

            # --- 2.1 Systemic Risk Check (CoVaR) ---
            # If the market is chaotic/volatile, we run a CoVaR check to ensure this bet doesn't correlate
            # dangerously with our existing portfolio (contagion risk).
            open_bets = context.get("open_bets", []) # Ensure pipeline populates this
            if open_bets:
                # Add current candidate to hypothetical portfolio
                hypothetical_portfolio = open_bets + [bet_candidate]
                covar_res = self.systemic_risk.measure(hypothetical_portfolio)

                # Check Delta CoVaR (Systemic contribution)
                if covar_res.get("delta_covar", 0) < -0.05: # High systemic risk contribution
                    decision.approved = False
                    decision.rejection_reason = f"Systemic Risk Veto: High Delta CoVaR ({covar_res['delta_covar']:.3f})"
                    return decision
                elif covar_res.get("delta_covar", 0) < -0.02:
                    decision.adjustments.append(f"Systemic Risk Warning: Moderate CoVaR ({covar_res['delta_covar']:.3f})")

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

        # --- 3.4.5 Smart Money Check (Financial Capability) ---
        # "Follow the Money"
        # We need european odds (from bet_candidate) and Asian (if available in context)
        euro_odds = {
            "home": bet_candidate.get("home_odds", 2.0),
            "draw": bet_candidate.get("draw_odds", 3.0),
            "away": bet_candidate.get("away_odds", 4.0),
            "opening_home": bet_candidate.get("opening_home_odds", 0.0) # Check if available
        }

        # Asian data usually needs a specialized provider, check context
        # For now, we simulate check if data exists
        asian_data = context.get("asian_markets", {}).get(match_id)

        sm_signal = self.smart_money.analyze(match_id, euro_odds, asian_data)

        if sm_signal.signal == "BEARISH":
            # Smart money disagrees with our selection (assuming we are bullish on Home)
            # Need to know WHICH side we are betting on.
            # Assuming 'selection' is in bet_candidate
            sel = bet_candidate.get("selection", "HOME").upper()

            # If SM is bearish on what we picked
            if sel == "HOME":
                # STRICT VETO: If Smart Money is fading us hard (> 0.7 strength)
                if sm_signal.strength > 0.7:
                    decision.approved = False
                    decision.rejection_reason = f"Smart Money Veto: Bearish Signal (Strength: {sm_signal.strength:.2f})"
                    return decision
                else:
                    # Weak bearish -> Penalty
                    decision.adjustments.append(f"Smart Money Bearish on HOME (Strength: {sm_signal.strength:.2f})")
                    bet_candidate["sm_multiplier"] = 0.5

        elif sm_signal.signal == "BULLISH":
            sel = bet_candidate.get("selection", "HOME").upper()
            if sel == "HOME":
                decision.adjustments.append(f"Smart Money Bullish on HOME (Strength: {sm_signal.strength:.2f})")
                bet_candidate["sm_multiplier"] = 1.2

        # --- 3.5 Game Theory Check (Strategic Defense) ---
        # Construct Payoff Matrix: Bettor (Rows) vs Market (Cols)
        # Actions: [Bet, Skip] vs [Stable, Drift Against]
        # Simplified payoff:
        #          Stable (+EV)   Drift (-EV)
        # Bet      +EV            -Stake
        # Skip     0              0

        try:
            ev = bet_candidate.get("ev", 0.05)
            stake = 1.0 # Normalized

            # Scenario A: Market stays stable, we win EV
            # Scenario B: Market drifts (smart money opposite), we lose stake (worst case) or just -EV
            # Let's assume Drift Against makes the bet -0.05 EV

            # Payoff Matrix (2x2)
            # Row 0 (Bet):  [+ev, -0.05]
            # Row 1 (Skip): [0,   0   ]

            payoff_matrix = np.array([
                [ev, -0.05],
                [0.0, 0.0]
            ])

            # Solve Nash
            gt_res = self.game_theory.solve_nash(payoff_matrix)

            if gt_res.is_solved:
                # Optimal strategy: Probability to BET (Action 0)
                prob_bet = gt_res.optimal_strategy[0]

                # If Nash says "Bet less than 50% of the time", it implies high risk of exploitation
                if prob_bet < 0.5:
                    decision.adjustments.append(f"Game Theory Warning: High Exploitation Risk (Optimal Bet Freq: {prob_bet:.2f})")
                    # We reduce stake by this probability factor
                    # e.g. if prob_bet is 0.3, we multiply stake by 0.3
                    # But we apply this later in modulation.
                    # Let's add it as a specific penalty adjustment string to be parsed or apply directly here?
                    # We'll apply it as a multiplier in step 5.
                    bet_candidate["gt_multiplier"] = prob_bet
                else:
                    bet_candidate["gt_multiplier"] = 1.0

        except Exception as e:
            logger.warning(f"Game Theory check failed: {e}")
            bet_candidate["gt_multiplier"] = 1.0

        # --- 3.6 Boardroom Meeting (NEW) ---
        # The Human-in-the-Loop Simulation.
        # CEO, CFO, and CTO debate the trade.
        board_ctx = {
            "ev": bet_candidate.get("ev", 0.0),
            "teleology_score": bet_candidate.get("teleology_score", 0.5),
            "drawdown": self.treasury.state.daily_pnl / max(self.treasury.state.total_capital, 1.0), # Approx
            "volatility": 0.05, # Should be fetched from VolatilityModulator
            "confidence": bet_candidate.get("confidence", 0.5),
            "entropy": 0.5 # Placeholder, should be in bet_candidate
        }

        board_decision = self.boardroom.convene(board_ctx)
        if not board_decision.approved:
            decision.approved = False
            decision.rejection_reason = "Boardroom Veto: " + "; ".join(board_decision.minutes)
            return decision

        # Apply Board Multiplier later
        decision.adjustments.append(f"Boardroom Multiplier: x{board_decision.final_multiplier:.2f}")
        bet_candidate["board_multiplier"] = board_decision.final_multiplier
        decision.rationale += "\nBoard Minutes:\n" + "\n".join(board_decision.minutes)

        # --- 3.7 Hawkes Momentum & Epistemic Uncertainty Checks ---
        epistemic_uncertainty = bet_candidate.get("epistemic_uncertainty", 0.5)
        # If Epistemic Uncertainty (I don't know) is high, we must penalize
        if epistemic_uncertainty > 0.8:
             decision.approved = False
             decision.rejection_reason = f"Epistemic Uncertainty Veto: Model is guessing ({epistemic_uncertainty:.2f})"
             return decision
        elif epistemic_uncertainty > 0.5:
             decision.adjustments.append(f"Epistemic Risk Warning: High uncertainty ({epistemic_uncertainty:.2f}). Stake reduced.")
             bet_candidate["epistemic_multiplier"] = 0.5
        else:
             bet_candidate["epistemic_multiplier"] = 1.0

        # Hawkes Momentum (Self-Exciting process)
        hawkes_home = bet_candidate.get("hawkes_home_intensity", 0.0)
        hawkes_away = bet_candidate.get("hawkes_away_intensity", 0.0)

        # If we are betting HOME and AWAY has massive Hawkes momentum, veto
        sel_for_hawkes = bet_candidate.get("selection", "HOME").upper()
        if sel_for_hawkes == "HOME" and hawkes_away > max(0.5, hawkes_home * 3):
             decision.approved = False
             decision.rejection_reason = f"Hawkes Momentum Veto: Away team is surging (λ_away={hawkes_away:.2f})"
             return decision
        elif sel_for_hawkes == "HOME" and hawkes_home > hawkes_away * 2:
             decision.adjustments.append(f"Hawkes Momentum Boost: Home team surging (λ_home={hawkes_home:.2f})")
             bet_candidate["hawkes_multiplier"] = 1.2
        else:
             bet_candidate["hawkes_multiplier"] = 1.0

        # --- 4. Kelly Sizing & Black-Litterman Sizing ---
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

        # Black-Litterman Single Asset Multiplier
        bl_multiplier = self.black_litterman.calculate_single_asset_multiplier(
            implied_prob=1.0 / max(bet_candidate.get("odds", 2.0), 1.01),
            model_prob=bet_candidate.get("prob_home", 0.5), # Assuming home
            epistemic_uncertainty=bet_candidate.get("epistemic_uncertainty", 0.5)
        )

        if bl_multiplier == 0.0:
            decision.approved = False
            decision.rejection_reason = "Black-Litterman Optimizer: Negative expected return post-blending."
            return decision

        if bl_multiplier != 1.0:
             final_stake_pct *= bl_multiplier
             decision.adjustments.append(f"Black-Litterman Conviction: x{bl_multiplier:.2f}")

        # --- 5. Physics Modulation ---
        # Apply advanced physics multipliers
        physics_ctx = self._extract_physics_context(match_id, context)
        phys_mult = self.physics_modulator.modulate(bet_candidate, physics_ctx)

        # Hypergraph Penalty Integration
        # If we had a structural warning, reduce stake by 20%
        if any("Hypergraph Warning" in adj for adj in decision.adjustments):
            phys_mult *= 0.8

        # EVT Penalty (from 0.6 above)
        if any("EVT Warning" in adj for adj in decision.adjustments):
            phys_mult *= 0.5

        # Game Theory Multiplier
        gt_mult = bet_candidate.get("gt_multiplier", 1.0)
        if gt_mult < 1.0:
            phys_mult *= gt_mult
            decision.adjustments.append(f"Game Theory Multiplier: x{gt_mult:.2f}")

        # Smart Money Multiplier
        sm_mult = bet_candidate.get("sm_multiplier", 1.0)
        if sm_mult != 1.0:
            phys_mult *= sm_mult
            decision.adjustments.append(f"Smart Money Multiplier: x{sm_mult:.2f}")

        # Board Multiplier
        board_mult = bet_candidate.get("board_multiplier", 1.0)
        if board_mult != 1.0:
            phys_mult *= board_mult

        # Epistemic Multiplier
        epi_mult = bet_candidate.get("epistemic_multiplier", 1.0)
        if epi_mult != 1.0:
            phys_mult *= epi_mult

        # Hawkes Multiplier
        hawkes_mult = bet_candidate.get("hawkes_multiplier", 1.0)
        if hawkes_mult != 1.0:
            phys_mult *= hawkes_mult

        if phys_mult <= 0.0:
            decision.approved = False
            decision.rejection_reason = "Physics/Board Kill Signal"
            return decision

        if phys_mult != 1.0:
            final_stake_pct *= phys_mult
            decision.adjustments.append(f"Physics/Board Multiplier: x{phys_mult:.2f}")

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

        # --- 6.1 Portfolio Stress Test (NEW) ---
        # Check if this bet, combined with current portfolio, violates VaR limits
        # We need current open bets. Assuming Treasury or a new 'PortfolioManager' has them.
        # For now, we simulate with a placeholder if not available in context.
        # Ideally, we should fetch: open_bets = self.treasury.get_open_bets()
        # Mocking retrieval for integration:
        open_bets = context.get("open_bets", []) # Need to ensure this is populated in Pipeline

        stress_bet = {
            "stake_amount": amount,
            "odds": bet_candidate.get("odds", 2.0),
            "prob_home": bet_candidate.get("prob_home", 0.5)
        }

        stress_res = self.stress_tester.check_portfolio_health(
            current_bets=open_bets,
            new_bet=stress_bet,
            total_capital=self.treasury.state.total_capital
        )

        if not stress_res["approved"]:
            decision.approved = False
            decision.rejection_reason = stress_res["reason"]
            return decision

        # Log stress if near limit
        if stress_res.get("var_pct", 0) > 0.15:
            decision.adjustments.append(f"Stress Warning: VaR at {stress_res['var_pct']:.1%}")


        # --- 6.5 Liquidity Check (NEW: LOB Simulation) ---
        # Ensure the amount doesn't exceed market depth and calculate realistic execution price
        league_name = bet_candidate.get("league", "Default")

        # 1. Max Safe Stake Check (Top level)
        max_safe_stake = self.liquidity.calculate_max_safe_stake(
            odds=bet_candidate.get("odds", 2.0),
            edge=bet_candidate.get("ev", 0.05),
            league=league_name
        )

        if amount > max_safe_stake:
            amount = max_safe_stake
            decision.adjustments.append(f"Liquidity Cap: Stake limited to {max_safe_stake:.2f} (Slippage protection)")

        # 2. Simulated Slippage Check
        # What is the effective price we get?
        exec_price, slippage_pct = self.liquidity.simulate_execution(amount, bet_candidate.get("odds", 2.0), league_name)

        # Re-check EV with execution price
        # New EV = Prob * ExecPrice - 1
        # Need correct probability for selection (default to 'prob_home' if not generic)
        selection_prob = bet_candidate.get("prob", bet_candidate.get("prob_home", 0.5))
        new_ev = selection_prob * exec_price - 1.0

        if new_ev <= 0:
            decision.approved = False
            decision.rejection_reason = f"Liquidity Veto: Slippage ({slippage_pct:.1%}) kills Edge. New EV: {new_ev:.2%}"
            return decision

        if slippage_pct > 0.05:
            decision.adjustments.append(f"Slippage Warning: {slippage_pct:.1%} (Price: {exec_price:.2f})")

        # 3. Optimal Execution Slicing for Large Stakes
        # If the amount is substantial, we don't execute at once.
        if amount > 5000.0:
            # Assume high urgency (0.8) for now, could be dynamic based on Edge decay
            schedule = self.optimal_execution.calculate_slicing_schedule(
                total_stake=amount,
                urgency=0.8,
                volatility=0.05,
                base_liquidity=self.liquidity.LEAGUE_LIQUIDITY.get(league_name, 5000.0)
            )
            if schedule.duration_steps > 1:
                decision.adjustments.append(f"Almgren-Chriss Execution: Sliced {amount:.2f} into {schedule.duration_steps} steps. (Est. Slip: {schedule.total_expected_slippage:.2%})")

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
        decision.rationale = f"Approved. Edge: {kelly_res.edge:.2%}. " + "; ".join(decision.adjustments) + f"\n\nBoard Minutes:\n" + "\n".join(board_decision.minutes)

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
