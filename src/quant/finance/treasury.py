"""
treasury.py – Advanced Capital Management (The CFO).

Kelly Criterion says "How much to bet", but Treasury says "Can we afford it?".
It manages liquidity, circuit breakers, and capital buckets.

Buckets:
  - Safe: Low-risk, steady yield (e.g. Arb/Value). 50% of capital.
  - Aggressive: High-risk, high-reward (e.g. Accumulators). 30%.
  - R&D: Experimental models. 20%.

Features:
  - Max Daily Drawdown: Stops trading if daily loss > 5%.
  - Liquidity Buffer: Keeps 10% cash at all times.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional

from loguru import logger
from src.system.config import settings
from src.system.container import container

@dataclass
class TreasuryState:
    total_capital: float = 10000.0
    daily_pnl: float = 0.0
    max_daily_drawdown_pct: float = 0.05 # 5% stop loss per day
    buckets: Dict[str, float] = field(default_factory=lambda: {
        "safe": 5000.0,
        "aggressive": 3000.0,
        "rnd": 2000.0
    })
    allocations: Dict[str, float] = field(default_factory=lambda: {
        "safe": 0.5,
        "aggressive": 0.3,
        "rnd": 0.2
    })
    locked_capital: float = 0.0 # Money in active bets

class TreasuryEngine:
    """
    Central Bank of the Betting System.
    """

    def __init__(self, state_path: Optional[Path] = None):
        self.state_path = state_path or settings.DATA_DIR / "treasury_state.json"
        self.state = TreasuryState()
        self.load_state()

    def get_sniper_stake(self) -> float:
        """
        Returns a fixed, small stake for high-speed Sniper entries.
        Bypasses standard approval but must respect absolute minimum liquidity.
        """
        # Fixed logic: 50 units or 1% of total capital, whichever is smaller
        stake = min(50.0, self.state.total_capital * 0.01)

        # Check absolute survival (must keep 10% min liquidity)
        available_liquid = self.state.total_capital - self.state.locked_capital
        min_liquidity = self.state.total_capital * 0.05
        if available_liquid - stake < min_liquidity:
            return 0.0

        return stake

    def get_auto_tuner_params(self) -> dict:
        """
        Fetch the optimal dynamic parameters like kelly_fraction from AutoTuner
        """
        auto_tuner = container.get("auto_tuner")
        if auto_tuner:
            return auto_tuner.get_current_params()
        return {"kelly_fraction": 0.25}


    def request_capital(self, amount: float, strategy_type: str = "safe") -> float:
        """
        Request capital for a bet.
        Returns the approved amount (may be less than requested or 0).
        """
        # 0. Apply AutoTuner dynamic limits
        tuner_params = self.get_auto_tuner_params()
        if tuner_params.get("kelly_fraction", 0.25) > 0.35:
            self.state.max_daily_drawdown_pct = 0.10
        else:
            self.state.max_daily_drawdown_pct = 0.05

        # 1. Circuit Breaker Check
        if self.state.daily_pnl < -(self.state.total_capital * self.state.max_daily_drawdown_pct):
            logger.warning("Treasury: Daily Drawdown Limit Hit. Capital request denied.")
            return 0.0

        # 2. Bucket Check
        bucket_cap = self.state.buckets.get(strategy_type, 0.0)
        if bucket_cap <= 0:
            # Fallback to safe if agg/rnd empty? No, strict buckets.
            logger.warning(f"Treasury: Bucket '{strategy_type}' is empty.")
            return 0.0

        # 3. Liquidity Check
        # Ensure we don't lock 100% of capital
        available_liquid = self.state.total_capital - self.state.locked_capital
        min_liquidity = self.state.total_capital * 0.05 # Keep 5% free

        max_approvable = available_liquid - min_liquidity

        approved = min(amount, bucket_cap, max_approvable)
        approved = max(0.0, approved)

        if approved > 0:
            # Lock capital
            self.state.buckets[strategy_type] -= approved
            self.state.locked_capital += approved
            self.save_state()
            logger.info(f"Treasury: Approved {approved:.2f} from '{strategy_type}'.")

        return approved

    def release_capital(self, amount: float, pnl: float, strategy_type: str = "safe"):
        """
        Release capital after a bet settles.
        """
        # Unlock the stake
        self.state.locked_capital -= amount
        # Ensure non-negative locked (floating point errors)
        self.state.locked_capital = max(0.0, self.state.locked_capital)

        # Update buckets and total
        # PnL is distributed: return stake to bucket + profit to bucket
        # Actually simple: bucket += stake + pnl

        self.state.buckets[strategy_type] += (amount + pnl)
        self.state.total_capital += pnl
        self.state.daily_pnl += pnl

        # Rebalance buckets if one grows too large?
        # For now, let winners run. But maybe periodic rebalance.

        self.save_state()
        logger.info(f"Treasury: Released {amount:.2f} + PnL {pnl:.2f} to '{strategy_type}'.")

    def rebalance_buckets(self, regime: str):
        """
        Dynamically adjust capital allocation based on Market Regime.
        Acting like a Central Bank changing reserve requirements.
        """
        regime = regime.lower()
        logger.info(f"Treasury: Rebalancing Buckets for Regime: {regime.upper()}")

        # Available liquid capital to redistribute (excluding locked)
        # We re-pool available funds and redistribute according to new ratios
        # But we must respect currently locked funds per bucket. This is complex.
        # Simplified approach: We change target allocations. Actual rebalancing happens over time?
        # No, for 'CRASH', we want to move AVAILABLE funds immediately to SAFE.

        # 1. Identify Target Allocations
        if regime in ["crash", "chaotic"]:
            new_alloc = {"safe": 1.0, "aggressive": 0.0, "rnd": 0.0}
        elif regime == "volatile":
            new_alloc = {"safe": 0.7, "aggressive": 0.2, "rnd": 0.1}
        elif regime == "stable":
            new_alloc = {"safe": 0.5, "aggressive": 0.3, "rnd": 0.2}
        else: # Normal default
            new_alloc = {"safe": 0.5, "aggressive": 0.3, "rnd": 0.2}

        self.state.allocations = new_alloc

        # 2. Redistribute Liquid Funds
        # Calculate total available liquid funds across all buckets
        total_available = sum(self.state.buckets.values()) # This is what's currently in buckets

        # Redistribute
        for bucket, ratio in new_alloc.items():
            self.state.buckets[bucket] = max(0.0, total_available * ratio)

        logger.success(f"Treasury: Rebalanced. New Buckets: {json.dumps(self.state.buckets, indent=2)}")
        self.save_state()

    def stress_test_portfolio(self, shock_factor: float = 0.20) -> Dict[str, Any]:
        """
        Simulates a market crash scenario (Stress Testing).
        Checks if the treasury can survive a sudden loss of `shock_factor`% of all open positions.

        Args:
            shock_factor: The percentage loss to simulate on locked capital (default 20%).

        Returns:
            A dict with solvency status and projected capital.
        """
        projected_loss = self.state.locked_capital * shock_factor
        projected_capital = self.state.total_capital - projected_loss

        # Determine survival
        # "Survival" means capital > 0 and liquidity > 5%
        is_solvent = projected_capital > 0
        liquidity_ratio = (projected_capital - (self.state.locked_capital * (1-shock_factor))) / projected_capital if projected_capital > 0 else 0.0

        status = "SOLVENT"
        if not is_solvent:
            status = "INSOLVENT"
        elif liquidity_ratio < 0.05:
            status = "ILLIQUID"

        return {
            "status": status,
            "shock_factor": shock_factor,
            "projected_loss": round(projected_loss, 2),
            "projected_capital": round(projected_capital, 2),
            "liquidity_ratio": round(liquidity_ratio, 3)
        }

    def reset_daily_pnl(self):
        """Call this at midnight."""
        logger.info(f"Treasury: Resetting Daily PnL (Was: {self.state.daily_pnl:.2f})")
        self.state.daily_pnl = 0.0
        self.save_state()

    def get_status(self) -> str:
        # Perform live stress test for the report
        stress = self.stress_test_portfolio(0.20)

        return (
            f"💰 Capital: {self.state.total_capital:.2f} | Locked: {self.state.locked_capital:.2f}\n"
            f"📉 Daily PnL: {self.state.daily_pnl:.2f} (Limit: {self.state.total_capital * -self.state.max_daily_drawdown_pct:.2f})\n"
            f"🛡️ Stress Test (20% Crash): {stress['status']} (Proj. Cap: {stress['projected_capital']:.2f})\n"
            f"🪣 Buckets: {json.dumps(self.state.buckets, indent=2)}"
        )

    def load_state(self):
        if self.state_path.exists():
            try:
                data = json.loads(self.state_path.read_text())
                self.state = TreasuryState(**data)
                logger.info("Treasury state loaded.")
            except Exception as e:
                logger.error(f"Failed to load treasury state: {e}")

    def save_state(self):
        try:
            data = self.state.__dict__
            self.state_path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.error(f"Failed to save treasury state: {e}")
