"""
liquidity_engine.py – Market Depth & Slippage Modeling (Simulated LOB).

Protects the "Edge" by ensuring stake sizes do not exceed market liquidity.
Large bets move odds (slippage), reducing EV. This engine simulates a Limit Order Book (LOB)
to estimate the execution price for a given stake size.

Concepts:
  - Limit Order Book: Stack of available liquidity at different price levels.
  - Market Impact: Walking the book (eating liquidity) moves the price against you.
  - Execution Price: Weighted average price of the filled order.

Usage:
    engine = LiquidityEngine()
    exec_price, slippage_pct = engine.simulate_execution(stake=5000, odds=2.0, league="Premier League")
"""
from __future__ import annotations

from loguru import logger
import math

class LiquidityEngine:
    """
    Estimates market liquidity and safe trade sizing using a simulated LOB.
    """

    # Estimated "Resistance" constants per league tier (Hypothetical Volume Multipliers)
    # Higher = More Liquid = Harder to move odds
    LEAGUE_LIQUIDITY = {
        "Premier League": 100_000.0,
        "La Liga": 80_000.0,
        "Serie A": 70_000.0,
        "Bundesliga": 75_000.0,
        "Ligue 1": 50_000.0,
        "Champions League": 150_000.0,
        "Championship": 20_000.0,
        "Super Lig": 15_000.0,
        "Eredivisie": 10_000.0,
        "Default": 5_000.0
    }

    def __init__(self):
        logger.debug("LiquidityEngine initialized (LOB Simulation Active).")

    def simulate_execution(self, stake: float, odds: float, league: str, match_minute: int = 0) -> tuple[float, float]:
        """
        Simulates "walking the book" to find the average execution price.

        Args:
            stake: Amount to bet.
            odds: Current top-of-book price.
            league: League name (defines depth profile).

        Returns:
            (execution_price, slippage_pct)
        """
        if stake <= 0:
            return odds, 0.0

        base_volume = self.LEAGUE_LIQUIDITY.get(league, self.LEAGUE_LIQUIDITY["Default"])
        if match_minute > 75:
            base_volume *= math.exp(-0.1 * (match_minute - 75))


        # LOB Model: Liquidity follows a power law or exponential decay away from the touch.
        # Simple Model: Linear density.
        # "Depth" = Amount available to bet before price moves 1 tick (e.g. 0.01).
        # Assume depth is proportional to base_volume.

        # Depth per tick (0.01 odds move)
        depth_per_tick = base_volume * 0.01 # 1% of volume available per tick? Heuristic.

        # Calculate how many ticks we eat through
        # Total Ticks = Stake / Depth_per_tick
        ticks_consumed = stake / depth_per_tick

        if ticks_consumed < 1.0:
            # Filled at top of book
            return odds, 0.0

        # Calculate weighted average price
        # Price P(v) = Odds - (v / Depth) * TickSize
        # Total Cost (in terms of potential payout units?) No, stake is constant.
        # We get less odds.
        # Avg Price approx = Odds - (TicksConsumed / 2) * TickSize

        tick_size = 0.01
        avg_price = odds - (ticks_consumed / 2.0) * tick_size

        # Cap slippage at reasonable bounds (e.g. 20%)
        min_price = odds * 0.8
        avg_price = max(avg_price, min_price)

        slippage_pct = (odds - avg_price) / odds

        return avg_price, slippage_pct

    def calculate_max_safe_stake(self, odds: float, edge: float, league: str, match_minute: int = 0) -> float:
        """
        Calculates the maximum stake that keeps execution price above Break-Even.
        Break-Even Price = 1 / Prob (where Edge = Prob*Odds - 1)

        We iterate or solve for Stake such that ExecutionPrice(Stake) > BreakEvenPrice.
        Or simpler: Target Slippage < Edge / 2 (Safety Margin).
        """
        target_slippage = edge * 0.5

        # Inverse of simulate_execution logic
        # Slippage = (TicksConsumed / 2) * TickSize / Odds
        # TicksConsumed = (Slippage * Odds * 2) / TickSize
        # Stake = TicksConsumed * DepthPerTick

        base_volume = self.LEAGUE_LIQUIDITY.get(league, self.LEAGUE_LIQUIDITY["Default"])
        if match_minute > 75:
            base_volume *= math.exp(-0.1 * (match_minute - 75))

        depth_per_tick = base_volume * 0.01
        tick_size = 0.01

        ticks_allowed = (target_slippage * odds * 2.0) / tick_size
        max_stake = ticks_allowed * depth_per_tick

        # Cap absolute max
        return min(max_stake, base_volume * 0.10) # Max 10% of volume

    def estimate_impact(self, stake: float, league: str) -> str:
        """Returns a human-readable impact assessment."""
        volume = self.LEAGUE_LIQUIDITY.get(league, self.LEAGUE_LIQUIDITY["Default"])
        ratio = stake / volume

        if ratio < 0.001:
            return "Invisible"
        if ratio < 0.01:
            return "Low Impact"
        if ratio < 0.05:
            return "Moderate Slip"
        return "Market Mover (High Slip)"
