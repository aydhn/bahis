"""
reflexivity_engine.py – The Soros Engine.

Implements the "Theory of Reflexivity", calculating a "Reflexivity Index"
based on odds momentum (1st derivative) and acceleration (2nd derivative).
It detects if a market is in a "Boom/Bust" reflexive loop (e.g., self-fulfilling
prophecy where dropping odds attract more money, dropping odds further).
"""

import numpy as np
from loguru import logger
from typing import List
from dataclasses import dataclass

@dataclass
class ReflexivityReport:
    index: float               # -1.0 to 1.0 (High positive means reflexive boom, negative means bust)
    momentum: float            # 1st derivative of odds
    acceleration: float        # 2nd derivative of odds
    is_reflexive: bool         # True if in a strong feedback loop
    signal: str                # "CONTRARIAN_FADE", "TREND_FOLLOW", "NEUTRAL"
    description: str


class ReflexivityEngine:
    """
    Analyzes odds movement to detect Soros-style reflexive feedback loops.
    """

    def __init__(self, momentum_threshold: float = 0.05, acceleration_threshold: float = 0.02):
        self.momentum_thresh = momentum_threshold
        self.accel_thresh = acceleration_threshold

    def analyze(self, odds_history: List[float], match_id: str = "Unknown") -> ReflexivityReport:
        """
        Analyzes a time series of odds for a specific selection.
        """
        if not odds_history or len(odds_history) < 3:
            return ReflexivityReport(0.0, 0.0, 0.0, False, "NEUTRAL", "Insufficient Data")

        try:
            odds_arr = np.array(odds_history)

            # Normalize to starting odds to calculate percentage moves
            start_odds = odds_arr[0]
            if start_odds == 0:
                start_odds = 1.0
            norm_odds = odds_arr / start_odds

            # Calculate 1st Derivative (Momentum - Velocity of odds drop/rise)
            # We look at the total change over the window
            momentum = norm_odds[-1] - norm_odds[0]

            # Calculate 2nd Derivative (Acceleration - Is the drop speeding up?)
            # Split history in half to see if second half dropped faster than first
            mid = len(norm_odds) // 2
            first_half_momentum = norm_odds[mid] - norm_odds[0]
            second_half_momentum = norm_odds[-1] - norm_odds[mid]

            acceleration = second_half_momentum - first_half_momentum

            # Calculate Reflexivity Index
            # If momentum and acceleration have the SAME sign, it's a compounding loop.
            # E.g. odds dropping (negative mom), and dropping FASTER (negative acc) -> Reflexive Boom for that team
            index = 0.0
            is_reflexive = False
            signal = "NEUTRAL"
            desc = "Market is stable."

            # We care about the magnitude
            combined_force = abs(momentum) + abs(acceleration)

            if np.sign(momentum) == np.sign(acceleration) and abs(momentum) > self.momentum_thresh:
                is_reflexive = True
                # Normalize index arbitrarily for reporting (-1 to 1)
                index = np.clip(combined_force * np.sign(momentum) * 5.0, -1.0, 1.0)

                if momentum < 0:
                    # Odds are crashing, and crashing faster -> Hype bubble
                    desc = f"Reflexive BOOM (Hype Bubble). Odds collapsing fast. Index: {index:.2f}"
                    if combined_force > (self.momentum_thresh + self.accel_thresh) * 1.5:
                        signal = "CONTRARIAN_FADE" # Bubble is too big, fade it
                        desc += " -> FADE THE PUBLIC."
                    else:
                        signal = "TREND_FOLLOW"
                else:
                    # Odds are drifting up, and drifting faster -> Abandonment
                    desc = f"Reflexive BUST (Abandonment). Odds drifting up fast. Index: {index:.2f}"
                    if combined_force > (self.momentum_thresh + self.accel_thresh) * 1.5:
                        signal = "CONTRARIAN_FADE" # Value might be hidden here now
                        desc += " -> VALUE REVERSAL."
                    else:
                        signal = "TREND_FOLLOW"

            return ReflexivityReport(
                index=float(index),
                momentum=float(momentum),
                acceleration=float(acceleration),
                is_reflexive=is_reflexive,
                signal=signal,
                description=desc
            )

        except Exception as e:
            logger.error(f"ReflexivityEngine error for {match_id}: {e}")
            return ReflexivityReport(0.0, 0.0, 0.0, False, "ERROR", str(e))
