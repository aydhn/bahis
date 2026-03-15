"""
philosophical_risk.py – Philosophical & Stoic Risk Assessor.

Combines statistical probabilities of extreme drawdowns with Stoic philosophy
to manage the psychological and mathematical risk of the algorithmic system.
Provides insights to the CEO Dashboard to temper irrational exuberance or panic.
"""
from dataclasses import dataclass
from typing import Dict, List
import numpy as np
from loguru import logger

@dataclass
class PhilosophicalInsight:
    state: str = "AEQUANIMITAS" # AEQUANIMITAS (Equanimity), HUBRIS (Overconfidence), AMOR_FATI (Embracing Chaos)
    stoic_quote: str = ""
    tail_risk_probability: float = 0.0
    suggested_kelly_penalty: float = 0.0
    narrative: str = ""

class PhilosophicalRiskEngine:
    """
    Evaluates market conditions and bankroll state to provide Stoic risk adjustments.
    """

    def __init__(self):
        logger.info("PhilosophicalRiskEngine initialized. Memento Mori.")
        self.quotes = {
            "AEQUANIMITAS": [
                "You have power over your mind - not outside events. Realize this, and you will find strength. - Marcus Aurelius",
                "He who fears death will never do anything worth of a man who is alive. - Seneca"
            ],
            "HUBRIS": [
                "To bear trials with a calm mind robs misfortune of its strength and burden. - Seneca",
                "No person has the power to have everything they want, but it is in their power not to want what they don't have. - Seneca"
            ],
            "AMOR_FATI": [
                "The impediment to action advances action. What stands in the way becomes the way. - Marcus Aurelius",
                "Fire tests gold, suffering tests brave men. - Seneca"
            ]
        }

    def assess_state(self, recent_pnl: List[float], win_rate: float, current_drawdown: float) -> PhilosophicalInsight:
        """
        Assess the current state combining statistical reality and philosophical grounding.

        Args:
            recent_pnl: List of recent PnL changes.
            win_rate: Current win rate of the system.
            current_drawdown: Current drawdown from peak (e.g., 0.15 for 15%).
        """
        insight = PhilosophicalInsight()

        # 1. Statistical Tail Risk (Probability of Ruin)
        # Simplified probability of extreme negative event based on recent variance
        if len(recent_pnl) > 5:
            std_dev = np.std(recent_pnl) if np.std(recent_pnl) > 0 else 1.0
            mean_pnl = np.mean(recent_pnl)
            # Z-score approximation for a 3-sigma drop
            tail_risk = 1.0 / (1.0 + np.exp((mean_pnl + 3*std_dev) / std_dev))
            insight.tail_risk_probability = float(tail_risk)
        else:
            insight.tail_risk_probability = 0.05

        # 2. Philosophical State Assignment
        if current_drawdown > 0.15:
            # Deep drawdown: We must embrace the chaos and stay the course rationally.
            insight.state = "AMOR_FATI"
            insight.suggested_kelly_penalty = 0.50 # Protect remaining capital, but don't stop completely
            insight.narrative = f"Drawdown at {current_drawdown:.1%}. Embrace the variance. System is being tested."
            insight.stoic_quote = np.random.choice(self.quotes["AMOR_FATI"])

        elif win_rate > 0.65 and mean_pnl > 0 and current_drawdown < 0.05:
            # Unusually high win rate: Danger of Hubris (Overconfidence)
            insight.state = "HUBRIS"
            # Mathematically penalize overconfidence to prevent mean-reversion blowup
            insight.suggested_kelly_penalty = 0.25
            insight.narrative = f"Win rate at {win_rate:.1%}. Beware of overconfidence. Mean reversion is inevitable."
            insight.stoic_quote = np.random.choice(self.quotes["HUBRIS"])

        else:
            # Normal operations
            insight.state = "AEQUANIMITAS"
            insight.suggested_kelly_penalty = 0.0
            insight.narrative = "Market conditions nominal. Maintain equanimity and execute edge."
            insight.stoic_quote = np.random.choice(self.quotes["AEQUANIMITAS"])

        logger.debug(f"Philosophical Assessment: {insight.state} | Penalty: {insight.suggested_kelly_penalty:.2f}")
        return insight
