"""
market_god.py – The Omniscient Market Strategist.

"I don't play the odds. I play the man."

This module aggregates high-level market signals (Regime, Smart Money, Game Theory)
into a single "God Signal". It detects if the "Fix is In" (Match Fixing),
if a "Black Swan" is imminent, or if the market is purely efficient (Nash Equilibrium).

It outputs a `GodSignal` that can veto or supercharge the entire pipeline.
"""
from dataclasses import dataclass
from typing import Dict, Any, Optional
from loguru import logger
import numpy as np

try:
    from src.extensions.regime_hmm import MarketRegimeHMM
except ImportError:
    MarketRegimeHMM = None

try:
    from src.extensions.smart_money import SmartMoneyDetector
except ImportError:
    SmartMoneyDetector = None

try:
    from src.quant.analysis.game_theory_engine import GameTheoryEngine
except ImportError:
    GameTheoryEngine = None

try:
    from src.extensions.behavioral_arbitrage import BehavioralArbitrage
except ImportError:
    BehavioralArbitrage = None

@dataclass
class GodSignal:
    """The divine verdict."""
    signal_type: str = "NEUTRAL" # BULLISH, BEARISH, NEUTRAL, BLACK_SWAN, FIX_DETECTED
    conviction: float = 0.0      # 0.0 - 1.0 (1.0 = Absolute Certainty)
    narrative: str = ""
    override_models: bool = False
    suggested_multiplier: float = 1.0

class MarketGod:
    """
    The Meta-Strategist that oversees the market state.
    """

    def __init__(self):
        self.hmm = MarketRegimeHMM() if MarketRegimeHMM else None
        self.smart_money = SmartMoneyDetector() if SmartMoneyDetector else None
        self.game_theory = GameTheoryEngine() if GameTheoryEngine else None
        self.behavioral_arb = BehavioralArbitrage() if BehavioralArbitrage else None
        logger.info("MarketGod initialized. Watching form the heavens.")

    def consult(self, match_id: str,
                odds_data: Dict[str, float],
                volatility_history: list[float] = None,
                model_prob: float = 0.5) -> GodSignal:
        """
        Consult the Market God for a verdict on a specific match.

        Args:
            match_id: Unique match identifier.
            odds_data: Current market odds {'home': 2.0, 'draw': 3.0, 'away': 4.0}.
            volatility_history: List of recent volatility metrics for Regime HMM.
        """
        signal = GodSignal()
        reasons = []
        score = 0.0 # -1.0 (Bearish) to +1.0 (Bullish) on Home

        # 1. Regime Analysis (HMM) - "The Weather"
        # If market is entering Chaos, God says "Stay Away" or "Fade".
        regime_risk = 0.0
        if self.hmm and volatility_history and len(volatility_history) > 10:
            pred = self.hmm.predict(np.array(volatility_history))
            # If chaotic probability is high
            if pred.next_state_probs[2] > 0.6: # High Chaos
                regime_risk = 1.0
                reasons.append("HMM Forecast: CHAOS imminent.")
            elif pred.next_state_probs[0] > 0.8: # Stable
                reasons.append("HMM Forecast: Stable Market.")

        # 2. Smart Money (Flow) - "The Whales"
        # We need to guess or have access to Asian lines.
        # For this high-level check, we assume we might have them in odds_data or skip.
        # Here we simulate a check if 'asian_home' key exists.
        sm_score = 0.0
        if self.smart_money and "asian_home" in odds_data:
            # Construct a minimal input for detector
            euro = {"home": odds_data.get("home", 0), "draw": odds_data.get("draw", 0), "away": odds_data.get("away", 0)}
            asian = {"line": odds_data.get("asian_line", 0), "home_odds": odds_data.get("asian_home", 0), "away_odds": 2.0}

            sm_res = self.smart_money.analyze(match_id, euro, asian)
            if sm_res.signal == "BULLISH":
                sm_score = 0.5 * sm_res.strength
                reasons.append(f"Smart Money Buying Home (Strength {sm_res.strength:.2f})")
            elif sm_res.signal == "BEARISH":
                sm_score = -0.5 * sm_res.strength
                reasons.append(f"Smart Money Selling Home (Strength {sm_res.strength:.2f})")

        # Behavioral Arbitrage (Sentiment/Recency Bias)
        arb_score = 0.0
        if self.behavioral_arb and "home" in odds_data:
            home_odds = odds_data.get("home", 2.0)
            arb_res = self.behavioral_arb.detect_mispricing(match_id, home_odds, model_prob)

            if arb_res.signal == "BEARISH":
                arb_score = -0.5 * arb_res.sentiment_score
                reasons.append(f"Behavioral Trap: Overvalued Fade ({arb_res.sentiment_score:.2f})")
            elif arb_res.signal == "BULLISH":
                arb_score = 0.5 * arb_res.sentiment_score
                reasons.append(f"Behavioral Value: Undervalued Edge ({arb_res.sentiment_score:.2f})")

        # 3. Game Theory (Strategy) - "The Trap"
        # Check if the market is perfectly efficient (Nash Equilibrium).
        # If implied prob == naive prob, edge is zero.
        # But if implied prob is WAY off, it's either Value or a Trap.
        # God logic: If Odds drop > 20% but no news, it's a Trap (Fix?).

        # Here we check for "Fix Detected" patterns (e.g. Draw odds < 1.8 in a competitive league)
        draw_odds = odds_data.get("draw", 3.0)
        if draw_odds < 1.90:
            # Biscuit Game signature
            signal.signal_type = "FIX_DETECTED"
            signal.conviction = 0.9
            signal.narrative = "Biscotti Game (Draw < 1.90). Mutually beneficial result likely."
            signal.override_models = True
            signal.suggested_multiplier = 0.0 # Or specific strategy for Draw
            # In God mode, we might suggest betting ON the fix?
            # Ethical constraint: We usually avoid or bet ON the fix if confident.
            # Let's say we override to bet Draw.
            return signal

        # Aggregate Scores
        total_score = sm_score + arb_score

        # Black Swan Check
        if regime_risk > 0.8:
            signal.signal_type = "BLACK_SWAN"
            signal.conviction = 0.8
            signal.narrative = "Market entering Chaos Regime. High Volatility expected."
            signal.override_models = True
            signal.suggested_multiplier = 0.0 # Cash is King
            return signal

        # Normal Signal
        if total_score > 0.3:
            signal.signal_type = "BULLISH"
            signal.conviction = min(total_score + 0.5, 1.0)
            signal.suggested_multiplier = 1.5
        elif total_score < -0.3:
            signal.signal_type = "BEARISH"
            signal.conviction = min(abs(total_score) + 0.5, 1.0)
            signal.suggested_multiplier = 0.5 # Fade or reduce stake

        signal.narrative = " | ".join(reasons) if reasons else "Market is Neutral."
        return signal
