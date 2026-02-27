"""
daily_briefing.py – Executive Daily Briefing (Yönetici Günlük Özeti).

Generates a CEO-level summary of the betting operations, covering:
- Global Market Regime (Chaos/Stable)
- Financial Health (Treasury)
- Top Teleological Opportunities (Narrative-driven)
- Hedge/Arbitrage Alerts

Usage:
    report = DailyBriefing.generate(context)
"""
from datetime import datetime
from typing import Dict, Any, List
from loguru import logger

class DailyBriefing:
    """
    Generates the 'Morning Briefing' text report.
    """

    @staticmethod
    def generate(context: Dict[str, Any]) -> str:
        """
        Constructs the briefing string from pipeline context.
        """
        date_str = datetime.now().strftime("%Y-%m-%d %H:%M")

        # 1. Global Macro
        treasury = context.get("treasury_status", "Unknown")
        regime = context.get("market_regime", "Normal")

        # 2. Intelligence (Top Picks)
        # Assuming 'ensemble_results' contains teleology info now
        picks = context.get("ensemble_results", [])
        # Filter for high confidence
        top_picks = sorted(picks, key=lambda x: x.get("confidence", 0) * x.get("ev", 0), reverse=True)[:3]

        # 3. Hedge/Arb
        # This would come from a dedicated scanner run or context
        arbs = context.get("arbitrage_opportunities", [])

        # Build Report
        report = [
            f"🌍 **EXECUTIVE BRIEFING** | {date_str}",
            "─────────────────────────────",
            f"**Market Regime:** {regime}",
            f"**Treasury:** {treasury}",
            "",
            "🧠 **INTELLIGENCE (Top 3):**"
        ]

        if not top_picks:
            report.append("  (No high-confidence signals today)")
        else:
            for i, p in enumerate(top_picks, 1):
                match = p.get("match_id", "?")
                sel = p.get("selection", "?")
                ev = p.get("ev", 0.0)
                conf = p.get("confidence", 0.0)
                narrative = p.get("teleology_narrative", "")

                line = f"{i}. **{match}** -> {sel} (EV: {ev:+.1%}, Conf: {conf:.0%})"
                report.append(line)
                if narrative:
                    report.append(f"   *\"{narrative}\"*")

        report.append("")
        report.append("⚡ **ACTIONS:**")

        if arbs:
            report.append(f"  🚨 {len(arbs)} Arbitrage opportunities detected!")
            for arb in arbs[:2]:
                report.append(f"  - ROI {arb['roi']:.1%}: {arb['home']['bookie']} vs {arb['away']['bookie']}")
        else:
            report.append("  (No active arbitrage or urgent hedges)")

        return "\n".join(report)
