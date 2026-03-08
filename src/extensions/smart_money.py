"""
smart_money.py – Smart Money Flow Detector.

"Follow the money."
This module detects informed trading activity by analyzing discrepancies
between the efficient Asian Handicap market and the retail 1X2 market.
It also flags "Steam Moves" (rapid, high-volume odds changes).

Signals:
    - BULLISH: Smart money is backing this selection.
    - BEARISH: Smart money is fading this selection.
    - NEUTRAL: No significant signal.

Techniques:
    - Asian vs European Discrepancy:
      If Asian Handicap implies 60% win prob, but 1X2 implies 55%,
      smart money is betting HEAVY on the favorite in the efficient market.
    - Steam Move:
      Odds drop > 5% in < 5 minutes → Institutional action.
    - Reverse Line Movement:
      Betting % is high on Team A (Public), but Line moves towards Team B (Sharps).
"""
from dataclasses import dataclass
from typing import Dict, Optional, List
import numpy as np
from loguru import logger

@dataclass
class SmartMoneySignal:
    signal: str = "NEUTRAL" # BULLISH, BEARISH, NEUTRAL
    strength: float = 0.0   # 0.0 to 1.0
    reason: str = ""
    asian_implied_prob: float = 0.0
    european_implied_prob: float = 0.0
    discrepancy: float = 0.0

class SmartMoneyDetector:
    """
    Detects institutional 'Smart Money' flows.
    """

    def __init__(self):
        self.history: Dict[str, List[float]] = {}
        logger.info("SmartMoneyDetector initialized.")

    def analyze(self,
                match_id: str,
                european_odds: Dict[str, float],
                asian_handicap: Optional[Dict[str, float]] = None,
                public_money_pct: Optional[float] = None) -> SmartMoneySignal:
        """
        Analyze a match for smart money signals.

        Args:
            match_id: Unique match ID.
            european_odds: {'home': 2.1, 'draw': 3.2, 'away': 3.5}
            asian_handicap: {'line': -0.5, 'home_odds': 1.95, 'away_odds': 1.90}
            public_money_pct: % of bets on Home (0.0 - 1.0), if available.
        """
        signal = SmartMoneySignal()

        # 1. European Implied Probability (Bookie Margin Removed)
        # Simple normalization
        h = european_odds.get("home", 0)
        d = european_odds.get("draw", 0)
        a = european_odds.get("away", 0)

        if h <= 1 or d <= 1 or a <= 1:
            return signal

        raw_probs = np.array([1/h, 1/d, 1/a])
        norm_probs = raw_probs / raw_probs.sum()
        prob_home_euro = norm_probs[0]
        signal.european_implied_prob = prob_home_euro

        # 2. Asian Handicap Analysis (The "Sharper" Market)
        if asian_handicap:
            line = asian_handicap.get("line", 0.0)
            ah_home = asian_handicap.get("home_odds", 2.0)
            ah_away = asian_handicap.get("away_odds", 2.0)

            # Convert AH to approximate Home Win Probability
            # Simplified model for -0.5 / -0.25 / 0 / +0.25 etc.
            # Using logistic function approximation or simple mapping
            # P(Win) ~ 1 / (1 + exp(-k * (rating_diff + line)))?
            # Better: Implied prob of AH line covering.

            # Basic conversion:
            # Implied AH Prob = 1 / ah_home (normalized)
            raw_ah = np.array([1/ah_home, 1/ah_away])
            norm_ah = raw_ah / raw_ah.sum()
            prob_home_ah_cover = norm_ah[0]

            # Adjust for the line to get pure "Win" probability comparison
            # If Line = -0.5, then AH Win = Home Win. Direct comparison.
            # If Line = 0 (Pk), then AH Win ~ Draw No Bet.
            # We need a robust converter. For MVP, we look for divergence in 'sentiment'.

            # Heuristic: If AH Odds for -0.5 are significantly lower than Euro Home Odds,
            # it means Asian market is more bullish on Home.

            # Let's focus on the -0.5 line which is equivalent to 1X2 Home Win
            if abs(line - (-0.5)) < 0.1:
                # Direct comparison possible
                signal.asian_implied_prob = prob_home_ah_cover
                discrepancy = prob_home_ah_cover - prob_home_euro
                signal.discrepancy = discrepancy

                if discrepancy > 0.05: # Asian market sees 5% higher prob
                    signal.signal = "BULLISH"
                    signal.strength = min(discrepancy * 10, 1.0)
                    signal.reason = f"Asian Market Bullish (-0.5 line implies {prob_home_ah_cover:.2f} vs Euro {prob_home_euro:.2f})"
                elif discrepancy < -0.05:
                    signal.signal = "BEARISH"
                    signal.strength = min(abs(discrepancy) * 10, 1.0)
                    signal.reason = "Asian Market Bearish"

        # 3. Reverse Line Movement (RLM)
        # If > 70% of public bets Home, but odds drift AGAINST Home (increase),
        # Smart money is on Away/Draw.
        if public_money_pct is not None:
            # Check odds movement (requires history, here we infer from current odds vs 'opening')
            # Assuming we might pass 'opening_odds' in european_odds dict for this check
            opening = european_odds.get("opening_home", h) # Default to current if no opening

            if public_money_pct > 0.70:
                if h > opening * 1.02: # Odds drifted up despite heavy betting
                    # Classic RLM
                    signal.signal = "BEARISH" # Smart money fading the public
                    signal.strength = 0.8
                    signal.reason = f"Reverse Line Movement: 70%+ Public on Home, but odds drifted {opening}->{h}"
            elif public_money_pct < 0.30:
                if h < opening * 0.98: # Odds dropped despite low betting
                    signal.signal = "BULLISH" # Smart money buying
                    signal.strength = 0.8
                    signal.reason = f"Reverse Line Movement: Low Public on Home, but odds dropped {opening}->{h}"

        return signal

    def detect_steam(self, match_id: str, current_odds: float, timestamp: float) -> Optional[Dict]:
        """
        Detects 'Steam Moves': Rapid, uniform odds movement across the market.
        Call this on every tick.
        """
        if match_id not in self.history:
            self.history[match_id] = []

        # Add new price
        self.history[match_id].append((timestamp, current_odds))

        # Keep last 5 minutes
        cutoff = timestamp - 300
        self.history[match_id] = [x for x in self.history[match_id] if x[0] > cutoff]

        if len(self.history[match_id]) < 2:
            return None

        start_price = self.history[match_id][0][1]
        pct_change = (current_odds - start_price) / start_price

        # Threshold: 5% drop in 5 mins
        if pct_change < -0.05:
            return {
                "type": "STEAM_BULLISH",
                "strength": abs(pct_change),
                "duration": timestamp - self.history[match_id][0][0],
                "reason": f"Steam Move: {start_price} -> {current_odds} in 5m"
            }
        elif pct_change > 0.05:
            return {
                "type": "STEAM_BEARISH",
                "strength": pct_change,
                "duration": timestamp - self.history[match_id][0][0],
                "reason": f"Drift: {start_price} -> {current_odds} in 5m"
            }

        return None
