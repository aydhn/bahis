"""
arbitrage_execution.py – High-Frequency Arbitrage Execution Manager.

This module automates the end-to-end execution of arbitrage opportunities
detected by the ArbitrageScanner. It ensures that cross-bookmaker stakes
are perfectly balanced for guaranteed profit while respecting liquidity
constraints and treasury limits.

Features:
  - Capital Allocation: Given an arb opportunity, decides the optimal total stake.
  - Stake Balancing: Distributes the total stake across the legs to equalize profit.
  - Liquidity Check: Validates that each leg can be absorbed by the market.
"""
from typing import Dict, Any, List, Optional
from loguru import logger
from dataclasses import dataclass

from src.system.container import container
from src.quant.finance.liquidity_engine import LiquidityEngine
from src.quant.finance.treasury import TreasuryEngine

@dataclass
class ArbLeg:
    selection: str
    odds: float
    bookie: str
    stake: float
    expected_return: float

@dataclass
class ArbExecutionPlan:
    match_id: str
    total_stake: float
    guaranteed_profit: float
    roi: float
    legs: List[ArbLeg]
    approved: bool
    rejection_reason: str = ""

class ArbitrageExecutionManager:
    """
    Manages the sizing and execution planning for Arbitrage opportunities.
    """

    def __init__(self):
        self.liquidity = LiquidityEngine()
        logger.info("ArbitrageExecutionManager initialized.")

    def _get_treasury(self) -> Optional[TreasuryEngine]:
        return container.get("treasury")

    def plan_execution(self, match_id: str, arb_signal: Dict[str, Any], max_total_stake: float = 1000.0) -> ArbExecutionPlan:
        """
        Creates a balanced betting plan for an arbitrage opportunity.

        Args:
            match_id: Unique match identifier.
            arb_signal: Output from ArbitrageScanner.scan()
            max_total_stake: Theoretical max we want to commit.

        Returns:
            ArbExecutionPlan with calculated stakes per leg.
        """
        if not arb_signal or arb_signal.get("type") != "ARBITRAGE":
            return ArbExecutionPlan(match_id, 0.0, 0.0, 0.0, [], False, "Invalid signal")

        implied_prob = arb_signal.get("implied_prob", 1.0)
        if implied_prob >= 1.0:
            return ArbExecutionPlan(match_id, 0.0, 0.0, 0.0, [], False, "Not an arbitrage (implied_prob >= 1.0)")

        # Extract odds
        try:
            home_odds = arb_signal.get("home", {}).get("odds")
            draw_odds = arb_signal.get("draw", {}).get("odds")
            away_odds = arb_signal.get("away", {}).get("odds")
            if home_odds is None or away_odds is None:
                raise KeyError("Missing home or away odds")
        except KeyError:
            return ArbExecutionPlan(match_id, 0.0, 0.0, 0.0, [], False, "Missing odds data in signal")

        # 1. Request capital from Treasury
        treasury = self._get_treasury()
        allocated_capital = max_total_stake
        if treasury:
            # We treat Arbitrage as "safe" bucket
            allocated_capital = treasury.request_capital(max_total_stake, strategy_type="safe")
            if allocated_capital <= 0:
                return ArbExecutionPlan(match_id, 0.0, 0.0, 0.0, [], False, "Treasury denied capital for Arb")

        # 2. Calculate ideal balanced stakes
        # Let T = total stake.
        # Stake_i = T / (odds_i * implied_prob)
        # This guarantees equal return on all outcomes: Return = T / implied_prob

        stake_home = allocated_capital / (home_odds * implied_prob)
        stake_away = allocated_capital / (away_odds * implied_prob)
        stake_draw = 0.0
        if draw_odds:
            stake_draw = allocated_capital / (draw_odds * implied_prob)

        # 3. Liquidity check
        # For simplicity, we assume standard league liquidity
        league = "Default"
        exec_home, slip_home = self.liquidity.simulate_execution(stake_home, home_odds, league)
        exec_away, slip_away = self.liquidity.simulate_execution(stake_away, away_odds, league)
        exec_draw, slip_draw = 0.0, 0.0
        if draw_odds:
            exec_draw, slip_draw = self.liquidity.simulate_execution(stake_draw, draw_odds, league)

        # If slippage ruins the arb, we must reject or scale down.
        # Let's recalculate implied prob with execution prices
        new_implied = (1.0 / exec_home) + (1.0 / exec_away)
        if exec_draw > 0:
            new_implied += (1.0 / exec_draw)

        # Ensure we still have an edge after slippage (implied_prob < 1.0)
        # Adding a small buffer (e.g. 0.999 instead of 1.0) to ensure actual profit after rounding
        if new_implied >= 0.995:
            # Release capital back if we reject
            if treasury:
                treasury.release_capital(allocated_capital, 0.0, strategy_type="safe")
            return ArbExecutionPlan(match_id, allocated_capital, 0.0, 0.0, [], False, "Liquidity slippage ruined the arb")

        # We are good! We will use the original stakes, but expect slightly lower returns due to slippage
        # In a perfect world we would iteratively solve for optimal stake that maximizes absolute profit,
        # but this is a solid heuristic.

        returns = [stake_home * exec_home, stake_away * exec_away]
        if exec_draw > 0:
            returns.append(stake_draw * exec_draw)

        guaranteed_return = min(returns)

        profit = guaranteed_return - allocated_capital
        roi = profit / allocated_capital if allocated_capital > 0 else 0.0

        legs = [
            ArbLeg("HOME", exec_home, arb_signal.get("home", {}).get("bookie", "unknown"), round(stake_home, 2), round(stake_home * exec_home, 2)),
            ArbLeg("AWAY", exec_away, arb_signal.get("away", {}).get("bookie", "unknown"), round(stake_away, 2), round(stake_away * exec_away, 2))
        ]

        if exec_draw > 0:
            legs.append(ArbLeg("DRAW", exec_draw, arb_signal.get("draw", {}).get("bookie", "unknown"), round(stake_draw, 2), round(stake_draw * exec_draw, 2)))

        logger.success(f"ArbitrageExecutionManager: Found executable arb for {match_id}. ROI: {roi:.4f}")

        return ArbExecutionPlan(
            match_id=match_id,
            total_stake=round(allocated_capital, 2),
            guaranteed_profit=round(profit, 2),
            roi=round(roi, 4),
            legs=legs,
            approved=True,
            rejection_reason=""
        )
