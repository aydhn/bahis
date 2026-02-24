from collections import deque
from typing import Dict, Any, List
from loguru import logger
import time

class CognitiveGuardian:
    """
    Philosophical & Psychological Safety Layer.
    Prevents 'Tilt', 'Chase', and 'Hubris'.
    'The market can remain irrational longer than you can remain solvent.'
    """

    def __init__(self):
        self.recent_bets = deque(maxlen=20)
        self.last_bet_time = 0
        self.consecutive_losses = 0
        self.consecutive_wins = 0
        self.locked_teams = {} # Team -> Unlock Time

    def record_bet(self, bet: Dict[str, Any]):
        """Call this when a bet is placed."""
        self.recent_bets.append(bet)
        self.last_bet_time = time.time()

        # Simple tracking (In real system, this would come from PnL feedback)
        # For now, we assume unbiased state.

    def record_outcome(self, pnl: float):
        """Update mental state based on outcome."""
        if pnl < 0:
            self.consecutive_losses += 1
            self.consecutive_wins = 0
        elif pnl > 0:
            self.consecutive_wins += 1
            self.consecutive_losses = 0

    def check_bet(self, bet_request: Dict[str, Any]) -> bool:
        """
        Analyze a proposed bet for cognitive biases.
        Returns True if safe, False if biased/dangerous.
        """
        stake = bet_request.get("stake", 0.0)
        team = bet_request.get("team", "Unknown")

        # 1. Tilt Protection (Stop Chasing)
        if self.consecutive_losses >= 3:
            # Check if stake is significantly higher than average of last 3
            avg_stake = self._get_recent_avg_stake(3)
            if stake > avg_stake * 1.5:
                logger.warning(f"CognitiveGuardian: BLOCKED bet on {team}. Reason: Chase Logic detected (Loss Streak + Increased Stake).")
                return False

        # 2. Hubris Protection (Overconfidence)
        if self.consecutive_wins >= 5:
            # Prevent "All In" mentality
            avg_stake = self._get_recent_avg_stake(5)
            if stake > avg_stake * 2.0:
                logger.warning(f"CognitiveGuardian: BLOCKED bet on {team}. Reason: Hubris detected (Win Streak + Excessive Stake).")
                return False

        # 3. Frequency Limit (Impulse Control)
        if time.time() - self.last_bet_time < 2.0: # 2 seconds
            logger.warning(f"CognitiveGuardian: BLOCKED bet on {team}. Reason: Impulse Control (Too fast).")
            return False

        return True

    def _get_recent_avg_stake(self, n: int) -> float:
        if not self.recent_bets:
            return 1.0

        stakes = [b.get("stake", 0.0) for b in list(self.recent_bets)[-n:]]
        if not stakes:
            return 1.0
        return sum(stakes) / len(stakes)
