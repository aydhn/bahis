"""
optimal_execution.py - Almgren-Chriss Inspired Optimal Execution Framework.

This module helps manage the execution of large stakes by breaking them down
into smaller slices over time. This minimizes market impact (slippage) at the
cost of exposing the execution to market risk (volatility).

Concepts:
  - Urgency: High urgency means executing faster (more slippage, less market risk).
  - Volatility: High volatility means executing faster to avoid adverse price movements.
  - Slicing Schedule: A discrete list of (time_delay, stake_amount) to execute.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
from loguru import logger

@dataclass
class ExecutionSlice:
    step: int
    stake_amount: float
    expected_slippage_pct: float

@dataclass
class ExecutionSchedule:
    total_stake: float
    slices: List[ExecutionSlice]
    total_expected_slippage: float
    duration_steps: int

class OptimalExecutionModel:
    """
    Implements a simplified discrete Almgren-Chriss execution algorithm.
    It calculates how to slice a large order to balance market impact and volatility risk.
    """

    def __init__(self, default_steps: int = 5):
        self.default_steps = default_steps
        logger.debug("OptimalExecutionModel initialized.")

    def calculate_slicing_schedule(self,
                                   total_stake: float,
                                   urgency: float = 0.5,
                                   volatility: float = 0.02,
                                   base_liquidity: float = 50000.0) -> ExecutionSchedule:
        """
        Calculates an execution schedule for a given total stake.

        Args:
            total_stake: Total amount to bet.
            urgency: Float between 0.1 (low) and 1.0 (high). Higher means faster execution.
            volatility: Expected market volatility (std dev of odds).
            base_liquidity: Estimated total available liquidity in the market.

        Returns:
            ExecutionSchedule object containing the slicing details.
        """
        # Ensure parameters are within reasonable bounds
        urgency = max(0.1, min(1.0, urgency))
        volatility = max(0.001, volatility)

        # Decide number of steps based on urgency and size
        # If urgency is 1.0, we just dump it all at once (1 step)
        # If urgency is low, we might take up to 10 steps
        if urgency > 0.9 or total_stake < base_liquidity * 0.01:
            steps = 1
        else:
            steps = max(2, int(self.default_steps * (1.1 - urgency)))

        # Simplified Almgren-Chriss trajectory (exponential decay)
        # We want to execute more early if urgency/volatility is high
        kappa = urgency * volatility * 100  # "Risk aversion" parameter

        times = np.linspace(0, 1, steps + 1)
        # If kappa is very small, this approaches a linear schedule (TWAP)
        if kappa < 1e-4:
            remaining = np.linspace(total_stake, 0, steps + 1)
        else:
            remaining = total_stake * (np.sinh(kappa * (1 - times)) / np.sinh(kappa))

        trade_sizes = -np.diff(remaining)

        # Correct for floating point rounding to ensure sum exactly matches total_stake
        trade_sizes = np.round(trade_sizes, 2)
        diff = total_stake - np.sum(trade_sizes)
        trade_sizes[0] += diff  # Add difference to first trade

        slices = []
        total_slippage = 0.0

        for i, size in enumerate(trade_sizes):
            # Simulate slippage for this slice.
            # Temporary impact is proportional to rate of trading (size / time_interval)
            # Time interval is 1/steps
            rate = size * steps
            impact_pct = (rate / base_liquidity) * 0.1  # Simplified impact model

            slices.append(ExecutionSlice(step=i+1, stake_amount=size, expected_slippage_pct=impact_pct))
            total_slippage += size * impact_pct

        # Weighted average slippage
        avg_slippage = total_slippage / total_stake if total_stake > 0 else 0.0

        return ExecutionSchedule(
            total_stake=total_stake,
            slices=slices,
            total_expected_slippage=avg_slippage,
            duration_steps=steps
        )
