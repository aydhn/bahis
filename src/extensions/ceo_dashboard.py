from src.quant.analysis.causal_reasoner import CausalReasoner
from src.extensions.philosophical_risk import PhilosophicalRiskEngine
"""
ceo_dashboard.py – "God Mode / CEO Vision" Aggregator.

This module acts as the central executive intelligence dashboard, aggregating signals
from Treasury, Risk Control Tower, Game Theory, and Physics to provide a high-level,
stoic strategic overview for the user assuming the roles of CEO/Fund Manager.
"""
from typing import Any, Optional
from loguru import logger

from src.quant.finance.treasury import TreasuryEngine
from src.core.boardroom import Boardroom


class CEODashboard:
    """Aggregates system vitals and generates a high-level strategic executive summary."""


    def __init__(self):
        self.treasury = TreasuryEngine()
        self.boardroom = Boardroom()

        try:
            self.philosophical_risk = PhilosophicalRiskEngine()
        except Exception:
            self.philosophical_risk = None

        try:
            self.causal_reasoner = CausalReasoner()
        except Exception:
            self.causal_reasoner = None

        logger.debug("CEODashboard (God Mode) initialized.")

    def enforce_strategic_vision(self, god_signal: Any):
        """
        Dynamically adjusts capital allocation based on the MarketGod signal.
        A true CEO shifts capital where the edge is, aggressively or defensively.
        """
        if not god_signal:
            return

        sig = god_signal.signal_type
        logger.info(f"CEODashboard: Enforcing strategic vision based on GodSignal: {sig}")

        # Re-allocate treasury buckets
        if sig == "BLACK_SWAN":
            self.treasury.rebalance_buckets("crash")
        elif sig == "CHAOTIC" or sig == "BEARISH":
            self.treasury.rebalance_buckets("volatile")
        elif sig == "BULLISH":
             # Extremely aggressive
             self.treasury.state.allocations = {"safe": 0.3, "aggressive": 0.6, "rnd": 0.1}
             self.treasury.save_state()
             self.treasury.rebalance_buckets("stable")
             logger.success("Treasury shifted to MAXIMUM AGGRESSION.")
        elif sig == "FIX_DETECTED":
             # High conviction, shift to safe/arb styles to exploit
             self.treasury.state.allocations = {"safe": 0.8, "aggressive": 0.2, "rnd": 0.0}
             self.treasury.save_state()
             self.treasury.rebalance_buckets("stable")


    def calculate_greeks(self) -> dict:
        """
        Calculate synthetic Greeks based on the current treasury state allocations.
        Delta = Net directional bias.
        Gamma = Convexity (how fast allocation changes).
        Vega = Sensitivity to Chaos (Regime).
        """
        if not hasattr(self, "treasury") or not self.treasury or not self.treasury.state:
            return {"delta": 0.0, "gamma": 0.0, "vega": 0.0}

        allocs = self.treasury.state.allocations
        safe = allocs.get("safe", 0.5)
        agg = allocs.get("aggressive", 0.3)
        rnd = allocs.get("rnd", 0.2)

        # Delta: Aggressive - Safe (Bias towards taking risk)
        delta = agg - safe

        # Gamma: High if aggressive > safe by a lot, indicating convex risk taking
        gamma = agg * 1.5 if agg > safe else agg * 0.5

        # Vega: RND allocation + safe baseline. High Vega means we are sensitive to exploring volatile markets
        vega = rnd * 2.0 + safe * 0.1

        return {
            "delta": delta,
            "gamma": gamma,
            "vega": vega
        }

    def generate_report(self, context: Optional[Any]) -> str:
        """
        Generates the executive summary report.
        """
        # Refresh treasury state
        self.treasury.load_state()
        state = self.treasury.state

        total_cap = state.total_capital
        daily_pnl = state.daily_pnl
        locked_cap = state.locked_capital
        pnl_emoji = "🟢" if daily_pnl >= 0 else "🔴"
        roi = (daily_pnl / total_cap) * 100 if total_cap > 0 else 0.0

        # Attempt to extract context info if available
        regime = "NORMAL (Assumed)"
        cycle = 0
        if context:
            if hasattr(context, 'cycle_id'):
                cycle = getattr(context, 'cycle_id', 0)
            if hasattr(context, 'ensemble_results') and context.ensemble_results:
                regime = context.ensemble_results[0].get("regime_status", regime)
            elif hasattr(context, 'volatility_reports') and context.volatility_reports:
                first_vol = next(iter(context.volatility_reports.values()), None)
                if first_vol:
                     regime = first_vol.regime

        # Simulate a quick Boardroom consensus for the overall market
        board_ctx = {
            "ev": 0.05,
            "teleology_score": 0.5,
            "drawdown": (locked_cap / total_cap) if total_cap > 0 else 0.0,
            "volatility": 0.05,
            "confidence": 0.5,
            "entropy": 0.5
        }
        board_decision = self.boardroom.convene(board_ctx)
        consensus_str = f"{board_decision.consensus_score:.2f}/1.00"

        # Philosophical Insight
        phil_state = "AEQUANIMITAS"
        quote = "“The obstacle is the way.” — Marcus Aurelius"
        if self.philosophical_risk:
             insight = self.philosophical_risk.assess_state([daily_pnl], 0.55, 0.0) # Mock current
             phil_state = insight.state
             if insight.stoic_quote: quote = insight.stoic_quote

        # Causal Insight (if available)
        causal_str = "No major causal anomalies detected."
        if self.causal_reasoner:
             causal_str = "Monitoring hidden confounders..."

        greeks = self.calculate_greeks()
        delta = greeks["delta"]
        gamma = greeks["gamma"]
        vega = greeks["vega"]

        report = (
            f"👁️ **GOD MODE: EXECUTIVE VISION** 👁️\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"⚡ **Cycle:** #{cycle} | **Regime:** {regime}\n\n"
            f"💼 **Treasury & Capital Management**\n"
            f"• **Total Bankroll:** {total_cap:.2f} ₺\n"
            f"• **Locked Capital:** {locked_cap:.2f} ₺\n"
            f"• **Daily PnL:** {pnl_emoji} {daily_pnl:+.2f} ₺ (ROI: {roi:+.2f}%)\n\n"
            f"🏛️ **Boardroom Sentinel Consensus**\n"
            f"• **Alignment:** {consensus_str} | **Status:** {'Active' if board_decision.approved else 'Bunker'}\n\n"
            f"📊 **Systemic Posture & Sensitivity (The Greeks)**\n"
            f"• **Delta (Directional):** {delta:+.2f}\n"
            f"• **Gamma (Convexity):** {gamma:+.2f}\n"
            f"• **Vega (Vol Sensitivity):** {vega:.2f}\n\n"
            f"🧠 **Philosophical State:** {phil_state}\n"
            f"🔗 **Causal Intel:** {causal_str}\n\n"
            f"_{quote}_"
        )
        return report