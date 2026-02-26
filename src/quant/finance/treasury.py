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

    def request_capital(self, amount: float, strategy_type: str = "safe") -> float:
        """
        Request capital for a bet.
        Returns the approved amount (may be less than requested or 0).
        """
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
        min_liquidity = self.state.total_capital * 0.10 # Keep 10% free

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

    def reset_daily_pnl(self):
        """Call this at midnight."""
        logger.info(f"Treasury: Resetting Daily PnL (Was: {self.state.daily_pnl:.2f})")
        self.state.daily_pnl = 0.0
        self.save_state()

    def get_status(self) -> str:
        return (
            f"💰 Capital: {self.state.total_capital:.2f} | Locked: {self.state.locked_capital:.2f}\n"
            f"📉 Daily PnL: {self.state.daily_pnl:.2f} (Limit: {self.state.total_capital * -self.state.max_daily_drawdown_pct:.2f})\n"
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
