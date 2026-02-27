"""
black_scholes_hedge.py – Black-Scholes-Merton for Sports Betting Hedging.

Treats an open bet as a European Binary (Digital) Option.
When we want to cash out or hedge, we evaluate if the bookie's current price
is fair compared to our theoretical Black-Scholes price, considering implied volatility.

Concepts:
  - S (Spot Price): Implied probability of the current odds.
  - K (Strike Price): 1.0 (binary win).
  - T (Time to Expiry): Remaining time in the match.
  - r (Risk-free rate): 0 (not relevant in 90 mins).
  - sigma (Volatility): Volatility of the odds/implied probability.
"""
from __future__ import annotations

import math
import numpy as np
from scipy.stats import norm
from loguru import logger
from typing import Dict, Any

class BlackScholesHedge:
    """
    Evaluates hedge/cashout opportunities using Black-Scholes options pricing
    adapted for binary sports betting markets.
    """

    def __init__(self):
        logger.debug("BlackScholesHedge engine initialized.")

    def calculate_binary_call(self, S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
        """
        Prices a Binary (Cash-or-Nothing) Call Option.
        In sports betting, a winning bet pays a fixed amount (odds * stake).

        S: Current implied probability (0 to 1)
        K: Target (usually 1.0, but we use a threshold approach for binary)
        T: Time remaining (in hours or days, normalized)
        sigma: Volatility
        r: Risk free rate

        Formula for Binary Call: e^{-rT} * N(d2)
        where d2 = (ln(S/K) + (r - sigma^2/2)*T) / (sigma * sqrt(T))
        """
        if T <= 0:
            return 1.0 if S >= K else 0.0

        if sigma <= 0:
            return 1.0 if S >= K else 0.0

        # Note: In sports, S is already a probability. We adapt the model slightly.
        # Let's map S to a price space, or use a simplified probabilistic drift.
        # For a true binary outcome, S itself is E[Payoff] if risk-neutral.
        # But if we have volatility (uncertainty of the drift), we use BSM.

        # Avoid log(0)
        S = max(S, 1e-5)
        K = max(K, 1e-5)

        d2 = (math.log(S / K) + (r - 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))

        return math.exp(-r * T) * norm.cdf(d2)

    def evaluate_cashout(self,
                         original_odds: float,
                         current_odds: float,
                         stake: float,
                         minutes_played: int,
                         volatility: float = 0.3) -> Dict[str, Any]:
        """
        Evaluates if a offered cashout or a hedge bet is mathematically +EV
        based on implied volatility.

        Args:
            original_odds: Odds when bet was placed.
            current_odds: Current market odds for the same selection.
            stake: Original stake amount.
            minutes_played: Current match minute (0-90).
            volatility: Implied volatility of the market (0.1 to 1.0).

        Returns:
            Dict with theoretical value and action recommendation.
        """
        if current_odds <= 1.0 or original_odds <= 1.0:
            return {"action": "HOLD", "reason": "Invalid odds."}

        # 1. Basic properties
        p_current = 1.0 / current_odds
        p_original = 1.0 / original_odds

        time_remaining = max(90 - minutes_played, 1)
        T_normalized = time_remaining / 90.0 # Time to expiry (0 to 1)

        # 2. Potential Payout
        payout = stake * original_odds

        # 3. Theoretical Value (BSM Binary Option)
        # We treat the current implied prob as Spot, and we want it to reach "1.0" (Win)
        # But standard BSM requires S and K. We'll use a modified approach:
        # Expected value is simply p_current * payout.
        # The option value adds a premium for volatility.

        # Actually, for a Binary Option, the price *is* the risk-neutral probability.
        # S = p_current. We don't use K=1.0 directly in log(S/K) because log(p/1) is negative.
        # We model the underlying "score state" instead, or just use the Option pricing as a risk-adjusted value.

        # Let's use a simpler heuristic for the Hedge Engine:
        # Fair Cashout Value = Expected Payout = p_current * payout
        fair_value = p_current * payout

        # Real bookie cashout is usually Fair Value * (1 - Margin)
        # We assume standard bookie cashout offer is 10% less than fair value.
        estimated_bookie_offer = fair_value * 0.90

        # But if volatility is HIGH, the value of HOLDING the option (Gamma/Vega) is higher.
        # So if Volatility > 0.5, we demand a higher cashout to let it go.
        volatility_premium = (volatility - 0.2) * 0.1 * fair_value if volatility > 0.2 else 0.0
        target_value = fair_value + volatility_premium

        # 4. Determine Action (Hedge / Stop Loss)
        # If current odds have drifted significantly higher, we might want to Stop Loss.
        action = "HOLD"
        reason = "Market is stable."

        if current_odds > original_odds * 1.5:
            action = "STOP_LOSS"
            reason = f"Odds drifted heavily ({original_odds} -> {current_odds}). Cut losses."
        elif current_odds < original_odds * 0.7:
            # We are winning. Should we Green Book?
            # Check if volatility is high. If high, holding is risky.
            if volatility > 0.4:
                action = "GREEN_BOOK"
                reason = "High volatility and favorable odds. Lock in profit."
            else:
                action = "HOLD_WINNING"
                reason = "Favorable odds, low volatility. Let it ride."

        return {
            "action": action,
            "reason": reason,
            "fair_value": round(fair_value, 2),
            "target_value": round(target_value, 2),
            "estimated_offer": round(estimated_bookie_offer, 2),
            "volatility_premium": round(volatility_premium, 2),
            "p_current": round(p_current, 3)
        }
