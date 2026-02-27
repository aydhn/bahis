"""
liquidity_engine.py – Market Depth & Slippage Modeling.

Protects the "Edge" by ensuring stake sizes do not exceed market liquidity.
Large bets move odds (slippage), reducing EV. This engine estimates
the maximum safe stake before slippage erodes the edge.

Concepts:
  - Slippage: The difference between expected price and executed price.
  - Market Depth: The volume available at the current price level.
  - Resilience: How quickly the market absorbs a large order.

Usage:
    engine = LiquidityEngine()
    max_stake = engine.calculate_max_stake(odds=2.0, edge=0.05, league="Premier League")
"""
from __future__ import annotations

from loguru import logger
import math

class LiquidityEngine:
    """
    Estimates market liquidity and safe trade sizing.
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
        logger.debug("LiquidityEngine initialized.")

    def calculate_slippage(self, stake: float, league: str, odds: float) -> float:
        """
        Estimates the percentage slippage for a given stake.
        Model: Slippage ~ (Stake / Volume) ^ k
        """
        volume = self.LEAGUE_LIQUIDITY.get(league, self.LEAGUE_LIQUIDITY["Default"])

        # Simple impact model: Linear impact approximation
        # Impact = k * (Stake / Volume)
        # k depends on odds (lower odds = higher volume usually, but higher sensitivity?)

        # Let's assume Volume is the amount matched at current price.
        # If Stake > 1% of Volume, we start seeing slippage.

        ratio = stake / volume

        # Heuristic: 1% of volume moves price by 0.1% ?
        # Slippage is roughly proportional to square root of stake in some models (square root law)
        # But here we stick to linear approximation for simplicity in sports

        if ratio < 0.001:
            return 0.0

        slippage_pct = ratio * 0.1 # 10% of volume = 1% price move
        return slippage_pct

    def calculate_max_safe_stake(self, odds: float, edge: float, league: str) -> float:
        """
        Calculates the maximum stake that keeps slippage below the Edge.
        If Slippage > Edge, the bet becomes -EV.

        We want: EV_after_slip > 0
        EV_raw = P * O - 1 = Edge
        O_slip = O * (1 - SlippagePct)
        EV_slip = P * O_slip - 1

        We simplify: Max allowed slippage = Edge / (Odds * Prob) ~= Edge / (1 + Edge) ~= Edge
        So we clamp SlippagePct < Edge * 0.5 (Safety Margin)
        """
        volume = self.LEAGUE_LIQUIDITY.get(league, self.LEAGUE_LIQUIDITY["Default"])

        # Target max slippage = 50% of the edge
        target_slippage = edge * 0.5

        # Slippage = (Stake / Volume) * 0.1
        # Stake = (Slippage / 0.1) * Volume

        max_stake = (target_slippage / 0.1) * volume

        # Cap absolute max to avoid "infinity" on high edges
        return min(max_stake, volume * 0.05) # Never bet more than 5% of estimated volume

    def estimate_impact(self, stake: float, league: str) -> str:
        """Returns a human-readable impact assessment."""
        volume = self.LEAGUE_LIQUIDITY.get(league, self.LEAGUE_LIQUIDITY["Default"])
        ratio = stake / volume

        if ratio < 0.001: return "Invisible"
        if ratio < 0.01: return "Low Impact"
        if ratio < 0.05: return "Moderate Slip"
        return "Market Mover (High Slip)"
