"""
system_architect.py – The Grand Strategist.

This module acts as the "CEO's Brain", setting the global strategic posture
based on financial health (Treasury), market conditions (Regime), and
macro sentiment. It moves beyond per-match analysis to system-wide
adaptation.

Directives:
- EXPANSION: Aggressive growth, high exposure allowed.
- CONSOLIDATION: Balanced approach, profit taking.
- BUNKER: Capital preservation, high conviction only.
- LIQUIDATION: Emergency risk reduction.
"""
from dataclasses import dataclass
from typing import Dict, Any, List
from loguru import logger

@dataclass
class StrategicDirective:
    """The output command from the Architect."""
    posture: str = "CONSOLIDATION"  # EXPANSION, CONSOLIDATION, BUNKER, LIQUIDATION
    max_daily_exposure: float = 0.10 # % of bankroll
    required_edge_multiplier: float = 1.0 # 1.0 = standard, 1.5 = high conviction
    allowed_leagues: List[str] = None # None = All
    rationale: str = ""

class SystemArchitect:
    """
    The High-Level Strategist.
    """

    def __init__(self):
        logger.info("SystemArchitect initialized. Ready to set global strategy.")

    def consult(self,
                treasury_status: Dict[str, Any],
                regime_metrics: Any,
                news_sentiment: float = 0.5) -> StrategicDirective:
        """
        Derive the strategic directive from current system state.

        Args:
            treasury_status: Dict from TreasuryEngine (drawdown, pnl, etc.)
            regime_metrics: RegimeMetrics object from MarketRegimeDetector.
            news_sentiment: Float 0.0-1.0 (0=Panic, 1=Euphoria).
        """
        directive = StrategicDirective()

        # Unpack inputs
        drawdown = treasury_status.get("drawdown", 0.0)
        regime = getattr(regime_metrics, "regime", "STABLE")

        reasons = []

        # 1. Treasury Logic (The CFO Voice)
        if drawdown > 0.15:
            directive.posture = "BUNKER"
            directive.max_daily_exposure = 0.02
            directive.required_edge_multiplier = 1.5
            reasons.append(f"High Drawdown ({drawdown:.1%})")
        elif drawdown > 0.05:
            directive.posture = "CONSOLIDATION"
            directive.max_daily_exposure = 0.05
            directive.required_edge_multiplier = 1.2
            reasons.append(f"Moderate Drawdown ({drawdown:.1%})")
        else:
            # Healthy Treasury
            directive.posture = "EXPANSION"
            directive.max_daily_exposure = 0.15
            directive.required_edge_multiplier = 1.0
            reasons.append("Healthy Treasury")

        # 2. Market Regime Logic (The Quant Voice)
        if regime == "CRASH":
            directive.posture = "LIQUIDATION"
            directive.max_daily_exposure = 0.0
            reasons.append("Market Crash Detected")
        elif regime == "CHAOTIC":
            directive.posture = "BUNKER"
            directive.max_daily_exposure = min(directive.max_daily_exposure, 0.01)
            directive.required_edge_multiplier = max(directive.required_edge_multiplier, 2.0)
            reasons.append("Chaotic Market Regime")
        elif regime == "VOLATILE":
            # If we were expanding, downgrade to consolidation
            if directive.posture == "EXPANSION":
                directive.posture = "CONSOLIDATION"
                directive.max_daily_exposure = 0.08
            reasons.append("High Volatility")

        # 3. Sentiment Logic (The Narrative Voice)
        # If sentiment is extreme panic (<0.2) or extreme euphoria (>0.8), be contrarian/cautious?
        # For now, simple safety check.
        if news_sentiment < 0.2:
            reasons.append("Macro Fear Dominates")
            if directive.posture == "EXPANSION":
                directive.posture = "CONSOLIDATION"

        directive.rationale = " | ".join(reasons)
        logger.info(f"Strategic Directive: {directive.posture} ({directive.rationale})")

        return directive
