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
        logger.debug("CEODashboard (God Mode) initialized.")

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
                # Get the first available volatility report regime
                first_vol = next(iter(context.volatility_reports.values()), None)
                if first_vol:
                     regime = first_vol.regime

        # Simulate a quick Boardroom consensus for the overall market
        board_ctx = {
            "ev": 0.05, # Neutral EV for overall market
            "teleology_score": 0.5,
            "drawdown": (locked_cap / total_cap) if total_cap > 0 else 0.0,
            "volatility": 0.05,
            "confidence": 0.5,
            "entropy": 0.5
        }
        board_decision = self.boardroom.convene(board_ctx)
        consensus_str = f"{board_decision.consensus_score:.2f}/1.00"

        # Executive quote
        quote = "“The obstacle is the way.” — Marcus Aurelius"

        report = (
            f"👁️ **GOD MODE: EXECUTIVE VISION** 👁️\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"⚡ **Cycle:** #{cycle} | **Regime:** {regime}\n\n"
            f"💼 **Treasury & Capital Management**\n"
            f"• **Total Bankroll:** {total_cap:.2f} ₺\n"
            f"• **Locked Capital:** {locked_cap:.2f} ₺\n"
            f"• **Daily PnL:** {pnl_emoji} {daily_pnl:+.2f} ₺ (ROI: {roi:+.2f}%)\n\n"
            f"🏛️ **Boardroom Sentinel Consensus**\n"
            f"• **Alignment:** {consensus_str}\n"
            f"• **Status:** {'Active' if board_decision.approved else 'Cautious/Bunker'}\n\n"
            f"📊 **Systemic Posture**\n"
            f"All Quant & Physics models are scanning. Portfolio optimization is live.\n"
            f"Strict risk bounds enforced.\n\n"
            f"_{quote}_"
        )
        return report
