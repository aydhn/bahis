from collections import deque
import numpy as np
from loguru import logger
from src.memory.db_manager import DBManager

class VolatilityModulator:
    """
    Adaptive Risk Scaling based on Market Volatility and Performance.
    'If the sea is rough, lower the sails.' - JP Morgan Approach.
    """

    def __init__(self, window_size: int = 50, target_vol: float = 0.05):
        self.db = DBManager()
        self.returns = deque(maxlen=window_size)
        self.outcomes = deque(maxlen=window_size) # 1 for Win, 0 for Loss
        self.target_vol = target_vol  # Target daily standard deviation (5%)
        self.current_vol = 0.0
        self.current_drawdown = 0.0
        self.current_win_rate = 0.0

        # Load history immediately
        self.load_history()

    def load_history(self):
        """Fetch PnL history from DB to initialize state."""
        try:
            # We need chronological PnL returns
            # Assuming 'bets' table has 'stake' and 'pnl'
            df = self.db.query("""
                SELECT stake, pnl, status
                FROM bets
                WHERE status IN ('won', 'lost')
                ORDER BY settled_at ASC
                LIMIT 200
            """)

            if df.is_empty():
                return

            # Calculate ROI per bet as a proxy for "Daily Return" if we bet daily
            # ROI = PnL / Stake
            pnl = df["pnl"].to_numpy()
            stake = df["stake"].to_numpy()

            # Avoid division by zero
            roi = np.divide(pnl, stake, out=np.zeros_like(pnl), where=stake!=0)

            self.returns.extend(roi)

            # Outcomes
            outcomes = (df["status"] == "won").cast(int).to_numpy()
            self.outcomes.extend(outcomes)

            self._recalculate()
            logger.info(f"VolatilityModulator initialized with {len(roi)} bets.")

        except Exception as e:
            logger.error(f"VolatilityModulator load error: {e}")

    def update_returns(self, pnl_percent: float, is_win: bool):
        """Add the latest PnL return (as a decimal, e.g. 0.02 for 2%)"""
        self.returns.append(pnl_percent)
        self.outcomes.append(1 if is_win else 0)
        self._recalculate()

    def _recalculate(self):
        """Calculate metrics: Volatility, Drawdown, Win Rate."""
        if len(self.returns) < 5:
            self.current_vol = self.target_vol
            return

        # 1. Volatility (Std Dev of Returns)
        self.current_vol = np.std(self.returns)

        # 2. Drawdown (Peak to Trough of Equity Curve)
        # Construct cumulative equity curve
        equity = np.cumsum(self.returns)
        peak = np.maximum.accumulate(equity)
        # Drawdown is negative distance from peak
        # Avoid division by zero if peak is 0 (start)
        # Simplified: Absolute drawdown amount
        dd = peak - equity
        self.current_drawdown = np.max(dd) if len(dd) > 0 else 0.0

        # 3. Win Rate (Last N bets)
        if self.outcomes:
            self.current_win_rate = np.mean(self.outcomes)

    def get_kelly_fraction(self) -> float:
        """
        Returns the safe Kelly Fraction (0.0 - 1.0).
        Aggressive: 1.0
        Normal: 0.5
        Defensive: 0.2
        Stop: 0.0
        """
        base_fraction = 0.5 # Default half-Kelly

        # 1. Volatility Adjustment
        # If Volatility is high (>10%), cut stake in half
        vol_scalar = 1.0
        if self.current_vol > 0.10:
            vol_scalar = 0.5
        elif self.current_vol > 0.05:
            vol_scalar = 0.8

        # 2. Drawdown Brake
        # If Drawdown > 20%, go to 'Survival Mode' (0.2x)
        dd_scalar = 1.0
        if self.current_drawdown > 0.20:
            dd_scalar = 0.2
        elif self.current_drawdown > 0.10:
            dd_scalar = 0.5

        # 3. Performance Boost
        # If Win Rate > 60%, can be slightly more aggressive
        perf_scalar = 1.0
        if self.current_win_rate > 0.60:
            perf_scalar = 1.2
        elif self.current_win_rate < 0.40:
            perf_scalar = 0.6

        final_fraction = base_fraction * vol_scalar * dd_scalar * perf_scalar

        # Cap at 1.0 (Full Kelly) and Floor at 0.1 (Minimum viable)
        return max(0.1, min(1.0, final_fraction))

    def get_status(self) -> str:
        """Return a human-readable status."""
        kf = self.get_kelly_fraction()

        regime = "NORMAL"
        if self.current_drawdown > 0.10:
            regime = "RECOVERY (Drawdown)"
        elif self.current_vol > 0.10:
            regime = "TURBULENT (High Vol)"
        elif self.current_win_rate > 0.60:
            regime = "BULL RUN (High Perf)"

        return f"{regime} | Kelly: {kf:.2f}x | Vol: {self.current_vol:.1%} | WR: {self.current_win_rate:.1%}"
