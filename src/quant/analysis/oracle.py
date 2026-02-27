"""
oracle.py – The Philosophical & Autonomous Synthesis Engine.

"The Oracle" does not just predict the score. It predicts the *meaning* of the match.
It synthesizes Physics, Finance, and Narrative into a single strategic directive.

Prophecy Structure:
- The Omen (Early Signal)
- The Reality (Data Analysis)
- The Verdict (Strategic Action)
"""
from typing import Dict, Any, List
from loguru import logger
from datetime import datetime
from src.quant.analysis.teleology import TeleologicalEngine

class TheOracle:
    """
    The High-Level Strategic Advisor.
    Combines:
    - Physics (Chaos, Entropy)
    - Finance (Treasury status, Market Regimes)
    - Narrative (News, Sentiment)
    - Teleology (Purpose & Motivation)
    """

    def __init__(self):
        self.teleology = TeleologicalEngine()

    def consult(self, context: Dict[str, Any]) -> str:
        """
        Generate a Daily Prophecy based on the current pipeline context.
        """
        physics = context.get("physics_reports", {})
        # Ensure treasury_status is a string for report generation if passed as dict
        treasury_status_raw = context.get("treasury_status")
        if isinstance(treasury_status_raw, dict):
            # Parse dict
            dd = treasury_status_raw.get("drawdown", 0.0)
            if dd > 0.15: treasury_status = "High Drawdown"
            elif dd > 0.05: treasury_status = "Drawdown"
            else: treasury_status = "Healthy"
        else:
            treasury_status = str(treasury_status_raw or "Healthy")

        market_regime = context.get("market_regime", "Normal")

        # 1. Analyze Chaos (The Omen)
        chaos_level = "Calm"
        chaos_reports = physics.get("chaos_reports", {})
        if chaos_reports:
            # Check average Lyapunov exponent
            lyaps = [r.params.max_lyapunov for r in chaos_reports.values() if hasattr(r, 'params')]
            if lyaps:
                avg_lyap = sum(lyaps) / len(lyaps)
                if avg_lyap > 0.05:
                    chaos_level = "Chaotic"
                elif avg_lyap > 0.01:
                    chaos_level = "Turbulent"

        # 2. Analyze Teleology (The Purpose)
        # Check if any "Purpose" driven matches exist
        teleology_insight = ""
        # context might contain matches, iterate to find interesting ones
        matches = context.get("matches")
        if matches is not None and not matches.is_empty():
             # Analyze just a sample or aggregate
             # For prophecy, we look for overall theme
             # Just running analysis on first match as proxy for now if individual results not available
             # Ideally context has pre-computed teleology for top matches
             pass

        # 3. Analyze Finance (The Reality)
        # Determine if we are in accumulation or preservation mode
        strategy = "Accumulate"
        if "Drawdown" in treasury_status or market_regime == "High Volatility":
            strategy = "Preserve"

        # 4. Generate The Verdict
        prophecy = f"🔮 **Oracle's Prophecy for {datetime.now().strftime('%Y-%m-%d')}**\n\n"

        if chaos_level == "Chaotic":
            prophecy += "🌪️ **The Omen:** The winds of chaos are howling. Randomness reigns.\n"
            prophecy += "🛡️ **The Strategy:** FORTRESS MODE. Avoid high-variance bets. Trust only significant edges.\n"
        elif chaos_level == "Turbulent":
            prophecy += "🌊 **The Omen:** The waters are choppy. Surprises lurk beneath.\n"
            prophecy += "⚖️ **The Strategy:** CAUTIOUS AGGRESSION. Reduce stake sizes. Look for value in mispriced favorites.\n"
        else:
            prophecy += "☀️ **The Omen:** The skies are clear. Logic dictates the outcome.\n"
            prophecy += "🚀 **The Strategy:** FULL STEAM AHEAD. Trust the models. Capitalize on clarity.\n"

        prophecy += f"\n💰 **Treasury Directive:** {strategy} Capital ({treasury_status}).\n"
        prophecy += f"📊 **Market Regime:** {market_regime}\n"

        prophecy += "\n🧘 **Philosophical Alignment:**\n"
        if strategy == "Preserve":
            prophecy += "*\"The first rule of compounding: Never interrupt it unnecessarily.\"* - Munger\n"
        else:
            prophecy += "*\"Fortune favors the bold, but history favors the prepared.\"*\n"

        return prophecy
