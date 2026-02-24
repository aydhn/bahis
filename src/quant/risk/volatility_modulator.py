from collections import deque
import numpy as np
from loguru import logger

class VolatilityModulator:
    """
    Adaptive Risk Scaling based on Market Volatility.
    'If the sea is rough, lower the sails.'
    """

    def __init__(self, window_size: int = 20, target_vol: float = 0.05):
        self.returns = deque(maxlen=window_size)
        self.target_vol = target_vol  # Target daily standard deviation (5%)
        self.current_vol = 0.0

    def update_returns(self, pnl_percent: float):
        """Add the latest PnL return (as a decimal, e.g. 0.02 for 2%)"""
        self.returns.append(pnl_percent)
        self._recalculate()

    def _recalculate(self):
        """Calculate rolling standard deviation."""
        if len(self.returns) < 5:
            self.current_vol = self.target_vol # Assume normal until enough data
            return

        self.current_vol = np.std(self.returns)

    def get_stake_multiplier(self) -> float:
        """
        Returns a multiplier (0.2 to 1.5) for the base stake.
        If current vol > target vol, multiplier < 1.0.
        """
        if self.current_vol == 0:
            return 1.0

        # Formula: Target Vol / Current Vol
        # Example: Target=5%, Current=10% -> Multiplier = 0.5
        ratio = self.target_vol / self.current_vol

        # Clamp between 0.2 (Safety floor) and 1.5 (Aggressive cap)
        multiplier = max(0.2, min(1.5, ratio))

        return multiplier

    def get_status(self) -> str:
        """Return a human-readable status of the volatility regime."""
        mult = self.get_stake_multiplier()
        if mult < 0.8:
            return "HIGH VOLATILITY (Defensive Mode)"
        elif mult > 1.2:
            return "LOW VOLATILITY (Aggressive Mode)"
        else:
            return "NORMAL REGIME"
