"""
microstructure_engine.py - Order Flow Imbalance (OFI) & VPIN Analyzer.

This module detects toxic order flow and market microstructure anomalies.
It computes:
  - VPIN (Volume-Synchronized Probability of Informed Trading): High values
    indicate adverse selection risk (smart money entering aggressively).
  - OFI (Order Flow Imbalance): Measures net buying pressure vs selling pressure
    at the top of the book.
"""
import numpy as np
from typing import List, Dict
from dataclasses import dataclass
from loguru import logger

@dataclass
class MicrostructureReport:
    vpin_score: float  # Range [0, 1]
    ofi_score: float   # Net volume imbalance
    is_toxic: bool     # True if VPIN > threshold or OFI is highly skewed
    signal: str        # "BULLISH", "BEARISH", "NEUTRAL", "TOXIC"

class MicrostructureEngine:
    """
    Computes microstructure metrics to detect informed trading and toxic flow.
    """
    def __init__(self, volume_bucket_size: float = 1000.0, vpin_threshold: float = 0.8):
        self.volume_bucket_size = volume_bucket_size
        self.vpin_threshold = vpin_threshold
        logger.debug(f"MicrostructureEngine initialized with bucket size {self.volume_bucket_size} and VPIN threshold {self.vpin_threshold}.")

    def compute_vpin(self, trade_volumes: List[float], price_changes: List[float]) -> float:
        """
        Calculates a simplified Volume-Synchronized Probability of Informed Trading (VPIN).
        In a real scenario, this would group trades into equal-volume buckets and calculate
        order imbalance within those buckets.

        Args:
            trade_volumes: List of trade sizes over a period.
            price_changes: Corresponding price change (tick direction) for each trade.
                           +1 for up-tick (buy initiated), -1 for down-tick (sell initiated), 0 for no change.
        Returns:
            VPIN score (0.0 to 1.0).
        """
        if not trade_volumes or not price_changes or len(trade_volumes) != len(price_changes):
            return 0.0

        # Group into volume buckets
        buckets = []
        current_bucket_vol = 0.0
        current_imbalance = 0.0

        for vol, direction in zip(trade_volumes, price_changes):
            current_bucket_vol += vol
            # We assume direction is already classified (+1 buy, -1 sell, 0 neutral)
            current_imbalance += vol * np.sign(direction)

            if current_bucket_vol >= self.volume_bucket_size:
                buckets.append({
                    "volume": current_bucket_vol,
                    "imbalance": abs(current_imbalance)
                })
                current_bucket_vol = 0.0
                current_imbalance = 0.0

        # Calculate VPIN over buckets
        if not buckets:
            return 0.0

        total_volume = sum(b["volume"] for b in buckets)
        total_imbalance = sum(b["imbalance"] for b in buckets)

        vpin = total_imbalance / total_volume if total_volume > 0 else 0.0
        return min(max(vpin, 0.0), 1.0) # Clamp between 0 and 1

    def compute_ofi(self, bid_sizes: List[float], ask_sizes: List[float],
                    bid_prices: List[float], ask_prices: List[float]) -> float:
        """
        Calculates Order Flow Imbalance (OFI) based on limit order book changes.
        Positive OFI indicates net buying pressure (bids increasing, asks decreasing).
        Negative OFI indicates net selling pressure.

        Args:
            bid_sizes, ask_sizes: Sizes at top of book over time.
            bid_prices, ask_prices: Prices at top of book over time.

        Returns:
            Net OFI score.
        """
        if len(bid_sizes) < 2 or len(ask_sizes) < 2:
            return 0.0

        total_ofi = 0.0

        # Calculate OFI as sum of changes in bid/ask depth
        for i in range(1, len(bid_sizes)):
            # Bid side contribution (e = \Delta V_bid if P_bid >= P_bid_prev else 0)
            delta_bid_vol = 0.0
            if bid_prices[i] >= bid_prices[i-1]:
                delta_bid_vol = bid_sizes[i] - (bid_sizes[i-1] if bid_prices[i] == bid_prices[i-1] else 0)

            # Ask side contribution (e = \Delta V_ask if P_ask <= P_ask_prev else 0)
            delta_ask_vol = 0.0
            if ask_prices[i] <= ask_prices[i-1]:
                delta_ask_vol = ask_sizes[i] - (ask_sizes[i-1] if ask_prices[i] == ask_prices[i-1] else 0)

            total_ofi += (delta_bid_vol - delta_ask_vol)

        return total_ofi

    def analyze(self, volume_data: Dict[str, List[float]], lob_data: Dict[str, List[float]]) -> MicrostructureReport:
        """
        Analyzes market microstructure using both VPIN and OFI.

        Args:
            volume_data: Dict containing 'volumes' and 'directions'.
            lob_data: Dict containing 'bid_sizes', 'ask_sizes', 'bid_prices', 'ask_prices'.

        Returns:
            MicrostructureReport object.
        """
        vpin = 0.0
        ofi = 0.0

        if volume_data:
            vpin = self.compute_vpin(volume_data.get('volumes', []), volume_data.get('directions', []))

        if lob_data:
            ofi = self.compute_ofi(
                lob_data.get('bid_sizes', []),
                lob_data.get('ask_sizes', []),
                lob_data.get('bid_prices', []),
                lob_data.get('ask_prices', [])
            )

        # Determine toxicity and signal
        is_toxic = vpin > self.vpin_threshold

        signal = "NEUTRAL"
        if is_toxic:
            signal = "TOXIC"
        elif ofi > 1000.0:  # Arbitrary threshold for significant buying pressure
            signal = "BULLISH"
        elif ofi < -1000.0: # Significant selling pressure
            signal = "BEARISH"

        return MicrostructureReport(
            vpin_score=vpin,
            ofi_score=ofi,
            is_toxic=is_toxic,
            signal=signal
        )
