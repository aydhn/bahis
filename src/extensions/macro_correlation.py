from loguru import logger
from typing import List, Dict, Any

class MacroCorrelationEngine:
    """
    MacroCorrelationEngine - Detects systemic risk across multiple bets.

    If the portfolio is heavily skewed in one direction (e.g., all favorites, all HOME),
    it imposes a severe penalty to prevent 'Black Swan' weekend wipeouts where
    the market narrative universally collapses.
    """

    def __init__(self, tilt_threshold: float = 0.75, min_sample: int = 4):
        self.tilt_threshold = tilt_threshold
        self.min_sample = min_sample

    def calculate_systemic_modifier(self, bets: List[Dict[str, Any]]) -> float:
        """
        Analyzes the list of bets and returns a multiplier [0.0 - 1.0].
        If > 75% are HOME, returns a strong penalty.
        """
        if len(bets) < self.min_sample:
            return 1.0

        home_count = sum(1 for b in bets if b.get('selection', '').upper() == 'HOME')
        away_count = sum(1 for b in bets if b.get('selection', '').upper() == 'AWAY')
        total = len(bets)

        home_tilt = home_count / total
        away_tilt = away_count / total

        max_tilt = max(home_tilt, away_tilt)
        dominant = "HOME" if home_tilt > away_tilt else "AWAY"

        if max_tilt > self.tilt_threshold:
            # E.g., if 100% tilt -> 0.3 multiplier (severe reduction)
            # If 75% tilt -> 0.7 multiplier (moderate reduction)
            excess = max_tilt - self.tilt_threshold  # Range [0.0, 0.25] for threshold=0.75
            max_excess = 1.0 - self.tilt_threshold
            penalty = 1.0 - (excess / max_excess) * 0.7  # Max penalty 70%

            modifier = max(0.3, penalty)
            logger.warning(
                f"[MacroCorrelation] Systemic tilt detected: {max_tilt:.0%} {dominant} bets! "
                f"Applying portfolio modifier: {modifier:.2f}x to protect capital."
            )
            return modifier

        return 1.0
