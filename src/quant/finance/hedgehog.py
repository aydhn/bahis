"""
hedgehog.py – Dynamic Hedging & Arbitrage Engine.

This module provides real-time financial hedging capabilities.
It calculates "Green Book" (guaranteed profit) opportunities and
dynamic Stop Loss levels for existing positions.

Concepts:
  - Green Book: Hedging a bet such that profit is guaranteed regardless of outcome.
  - Hedge Ratio: Optimal stake for the hedge bet.
  - Stop Loss: Selling out of a position to minimize loss when odds drift against us.
  - Arbitrage: Exploiting price differences between bookmakers (theoretical here).
  - Dynamic Hedging: Using VolatilityModulator to adjust hedge thresholds.
"""
from typing import Dict, Optional, Any
from src.quant.finance.black_scholes_hedge import BlackScholesHedge

class ArbitrageScanner:
    """
    Scans for Arbitrage opportunities across different bookmakers.
    Requires odds data from multiple sources.
    """
    def __init__(self):
        pass

    def scan(self, match_odds: Dict[str, Dict[str, float]]) -> Optional[Dict[str, Any]]:
        """
        Detects arbitrage in a single match given odds from multiple bookies.

        Args:
            match_odds: {
                "bookie_A": {"home": 2.1, "draw": 3.2, "away": 3.5},
                "bookie_B": {"home": 2.0, "draw": 3.4, "away": 3.6},
                ...
            }

        Returns:
            Arb opportunity dict or None.
        """
        if not match_odds or len(match_odds) < 2:
            return None

        # Find best odds for each outcome
        best_home = (0.0, "")
        best_draw = (0.0, "")
        best_away = (0.0, "")

        for bookie, odds in match_odds.items():
            if odds.get("home", 0) > best_home[0]: best_home = (odds["home"], bookie)
            if odds.get("draw", 0) > best_draw[0]: best_draw = (odds["draw"], bookie)
            if odds.get("away", 0) > best_away[0]: best_away = (odds["away"], bookie)

        # Calculate Implied Probability
        if best_home[0] == 0 or best_draw[0] == 0 or best_away[0] == 0:
            return None

        implied_prob = (1.0 / best_home[0]) + (1.0 / best_draw[0]) + (1.0 / best_away[0])

        if implied_prob < 1.0:
            roi = (1.0 / implied_prob) - 1.0
            return {
                "type": "ARBITRAGE",
                "roi": roi,
                "home": {"odds": best_home[0], "bookie": best_home[1]},
                "draw": {"odds": best_draw[0], "bookie": best_draw[1]},
                "away": {"odds": best_away[0], "bookie": best_away[1]},
                "implied_prob": implied_prob
            }

        return None


class HedgeHog:
    """
    Real-time hedging calculator.
    """

    def __init__(self):
        self.arb_scanner = ArbitrageScanner()
        self.bsm_hedge = BlackScholesHedge()

    def calculate_green_book(self,
                             current_stake: float,
                             current_odds: float,
                             hedge_odds: float,
                             commission: float = 0.0) -> Dict[str, float]:
        """
        Calculates the stake required to 'Green Book' (guarantee profit).

        Scenario: We backed Home @ 2.0. Now Home @ 1.5 (Lay or Back Away?).
        If we are backing the opposite outcome (e.g. Double Chance X2), we need
        hedge_odds for that outcome.

        Args:
            current_stake: The stake of the existing bet.
            current_odds: The odds the bet was placed at (Back odds).
            hedge_odds: The current odds available for the OPPOSING outcome (e.g. Lay Home or Back Draw/Away).
            commission: Exchange commission (e.g. 0.05 for Betfair).

        Returns:
            Dict with 'hedge_stake', 'guaranteed_profit', 'roi'.
        """
        # Cashout Value = (Stake * OriginalOdds) / CurrentOdds (if backing same selection)
        # But here we assume hedge_odds is for the COUNTER outcome (Lay or X2).

        # Proper Green Book formula if using Lay (Exchange):
        # Lay Stake = (Back Stake * Back Odds) / Lay Odds
        # Profit = Lay Stake - Back Stake (minus commission)

        if hedge_odds <= 1.0:
            return {"error": "Invalid hedge odds"}

        # Simplified "Cashout" logic using implied probabilities if we treat hedge_odds as the market price to EXIT.
        # If hedge_odds is the Back odds of the Counterpart:
        # We need to bet on Counterpart such that Profit_Home = Profit_Counterpart.
        # Profit_Home = Stake*Odds - Stake - HedgeStake
        # Profit_Counter = HedgeStake*HedgeOdds - HedgeStake - Stake
        # ... this algebra gets complex for 3-way.

        # Let's stick to the "Cash Out" value approximation which is standard.
        # Cashout = Stake * (OriginalOdds / CurrentOddsForSameSelection)
        # To support "Green Book", we need the Lay odds of the same selection.
        # Let's assume 'hedge_odds' IS the Lay odds (or equivalent exit price).

        hedge_stake = (current_stake * current_odds) / hedge_odds
        potential_profit = hedge_stake - current_stake

        return {
            "hedge_stake": round(hedge_stake, 2),
            "guaranteed_profit": round(potential_profit, 2),
            "roi": round(potential_profit / current_stake, 4)
        }

    def check_hedge_opportunity(self,
                                position: Dict[str, Any],
                                live_odds: float) -> Optional[Dict[str, Any]]:
        """
        Checks if a position should be hedged based on live odds.

        Args:
            position: {'match_id': '...', 'selection': 'HOME', 'stake': 100, 'odds': 2.0}
            live_odds: Current market odds for the same selection (Back).
                       Note: To hedge, we usually need Lay odds.
                       Approximation: Lay Odds ~= Back Odds * 1.02 (spread).

        Returns:
            Hedge signal dict or None.
        """
        original_odds = position.get("odds", 0.0)
        stake = position.get("stake", 0.0)

        if original_odds <= 1.0 or live_odds <= 1.0:
            return None

        # Estimate Lay Odds (Counterpart)
        lay_odds = live_odds * 1.05 # Spread approximation

        # 1. Profit Taking (Green Book)
        # If odds dropped > 20%
        if live_odds < original_odds * 0.8:
            res = self.calculate_green_book(stake, original_odds, lay_odds)

            if res.get("guaranteed_profit", 0) > stake * 0.1: # Min 10% profit
                return {
                    "action": "HEDGE_PROFIT",
                    "reason": f"Odds dropped {original_odds}->{live_odds}",
                    "details": res
                }

        # 2. Stop Loss
        # If odds drifted > 30% (e.g. 2.0 -> 2.6)
        if live_odds > original_odds * 1.3:
            # Cashout Value = (Stake * Orig) / Current
            cashout_value = (stake * original_odds) / live_odds
            loss = stake - cashout_value

            return {
                "action": "STOP_LOSS",
                "reason": f"Odds drifted {original_odds}->{live_odds}",
                "loss": round(loss, 2),
                "cashout_value": round(cashout_value, 2)
            }

        return None

    def dynamic_hedge(self,
                      position: Dict[str, Any],
                      live_odds: float,
                      volatility: float) -> Optional[Dict[str, Any]]:
        """
        Advanced hedge check that adapts thresholds based on market volatility.
        High volatility -> Wider stops (avoid noise).
        Low volatility -> Tighter stops.

        Args:
            volatility: GARCH volatility or std dev of odds (0.01 - 0.10 range typical)
        """
        # Base thresholds
        stop_loss_pct = 0.30 # 30% drift
        take_profit_pct = 0.20 # 20% drop

        # Adjust by volatility
        # If vol is high (0.05+), widen stops to 1.5x
        vol_factor = 1.0 + max(0, (volatility - 0.02) * 10) # e.g. 0.05 -> 1.3

        adj_stop_loss = stop_loss_pct * vol_factor
        adj_take_profit = take_profit_pct # Keep profit target static or tighten? Let's keep static for now.

        original_odds = position.get("odds", 0.0)

        # Check logic with adjusted thresholds
        # Let's leverage the Black-Scholes module for a more advanced evaluation
        # Assuming we know current match minute. If not, use 45 as a dummy mid-point
        current_minute = position.get("current_minute", 45)

        bsm_eval = self.bsm_hedge.evaluate_cashout(
            original_odds=original_odds,
            current_odds=live_odds,
            stake=position.get("stake", 0.0),
            minutes_played=current_minute,
            volatility=volatility
        )

        action = bsm_eval["action"]
        if action in ["STOP_LOSS", "GREEN_BOOK"]:
            return {
                "action": action,
                "reason": bsm_eval["reason"] + f" (BSM Target: {bsm_eval['target_value']})",
                "details": bsm_eval
            }

        # Fallback to standard check if BSM is neutral
        if live_odds > original_odds * (1.0 + adj_stop_loss):
             return self.check_hedge_opportunity(position, live_odds)

        if live_odds < original_odds * (1.0 - adj_take_profit):
             return self.check_hedge_opportunity(position, live_odds)

        return None
