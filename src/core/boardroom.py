"""
boardroom.py – Multi-Agent Corporate Board Simulation.

Simulates a corporate board meeting where specialized agents (CEO, CFO, CTO)
debate each decision. This adds a "Human-in-the-loop" simulation layer
to the risk management process.

Agents:
  - CEO (Chief Executive Officer): Focuses on Growth, Momentum, and Narrative.
  - CFO (Chief Financial Officer): Focuses on Risk, Drawdown, and Solvency.
  - CTO (Chief Technology Officer): Focuses on Model Confidence, Data Quality, and Entropy.
"""
from dataclasses import dataclass
from typing import Dict, Any, List
import numpy as np

@dataclass
class BoardVote:
    role: str
    stake_multiplier: float # 0.0 to 2.0 (1.0 = Neutral)
    rationale: str

@dataclass
class BoardDecision:
    approved: bool
    final_multiplier: float
    minutes: List[str]
    consensus_score: float # 0.0 to 1.0 (Agreement level)

class Boardroom:
    """
    The decision-making body.
    """

    class CEO:
        """Growth & Vision."""
        def evaluate(self, context: Dict[str, Any]) -> BoardVote:
            # CEO likes high EV and Momentum
            ev = context.get("ev", 0.0)
            teleology = context.get("teleology_score", 0.5)

            mult = 1.0
            reasons = []

            if ev > 0.10:
                mult += 0.2
                reasons.append("High EV opportunity")

            if teleology > 0.7:
                mult += 0.1
                reasons.append("Strong narrative momentum")

            if ev < 0.02:
                mult -= 0.2
                reasons.append("Growth potential too low")

            return BoardVote("CEO", mult, "; ".join(reasons) or "Neutral")

    class CFO:
        """Risk & Solvency."""
        def evaluate(self, context: Dict[str, Any]) -> BoardVote:
            # CFO hates volatility and drawdown
            drawdown = context.get("drawdown", 0.0)
            volatility = context.get("volatility", 0.05)

            mult = 1.0
            reasons = []

            if drawdown > 0.10:
                mult *= 0.5
                reasons.append(f"Significant Drawdown ({drawdown:.1%}). Cutting exposure")

            if volatility > 0.08:
                mult *= 0.8
                reasons.append("Market volatility high")

            if drawdown < 0.02 and volatility < 0.04:
                mult *= 1.1
                reasons.append("Balance sheet is healthy")

            return BoardVote("CFO", mult, "; ".join(reasons) or "Neutral")

    class CTO:
        """Technology & Precision."""
        def evaluate(self, context: Dict[str, Any]) -> BoardVote:
            # CTO likes high confidence and low entropy
            confidence = context.get("confidence", 0.0)
            entropy = context.get("entropy", 0.5)

            mult = 1.0
            reasons = []

            if confidence > 0.75:
                mult += 0.15
                reasons.append(f"High Model Confidence ({confidence:.0%})")

            if entropy > 0.8: # High uncertainty
                mult *= 0.6
                reasons.append("High System Entropy. Predicting noise")

            return BoardVote("CTO", mult, "; ".join(reasons) or "Neutral")

    def __init__(self):
        self.ceo = self.CEO()
        self.cfo = self.CFO()
        self.cto = self.CTO()

    def convene(self, context: Dict[str, Any]) -> BoardDecision:
        """
        Calls the board meeting. Aggregates votes.
        """
        votes = [
            self.ceo.evaluate(context),
            self.cfo.evaluate(context),
            self.cto.evaluate(context)
        ]

        # Calculate Consensus
        multipliers = [v.stake_multiplier for v in votes]
        avg_mult = sum(multipliers) / len(multipliers)

        # Minutes
        minutes = [f"{v.role}: {v.rationale} (x{v.stake_multiplier:.2f})" for v in votes]
        minutes.append(f"**Final Board Decision**: x{avg_mult:.2f} Stake Multiplier")

        # Consensus Score (inverse of variance)
        std_dev = np.std(multipliers)
        consensus = max(0.0, 1.0 - std_dev)

        return BoardDecision(
            approved=(avg_mult > 0.0),
            final_multiplier=avg_mult,
            minutes=minutes,
            consensus_score=consensus
        )
