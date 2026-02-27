"""
arbitrage_scanner.py – Arbitrage & Surebet Detection Engine.

Scans for guaranteed profit opportunities (Surebets) across different bookmakers
or within the same bookmaker due to pricing inefficiencies.

Supports:
- 2-way Surebets (e.g., Tennis, Basketball, or Football Over/Under)
- 3-way Surebets (e.g., Football 1X2)
"""
from typing import Dict, Any, List, Optional

class ArbitrageScanner:
    """
    Detects arbitrage opportunities in odds data.
    """

    def scan(self, match_odds: Dict[str, Dict[str, float]]) -> Optional[Dict[str, Any]]:
        """
        Scans for 3-way arbitrage (1X2) given a dictionary of bookmaker odds.

        Args:
            match_odds: {
                "bookie_A": {"home": 2.1, "draw": 3.2, "away": 3.5},
                "bookie_B": {"home": 2.0, "draw": 3.4, "away": 3.6},
                ...
            }

        Returns:
            Dict containing arbitrage details if found, else None.
        """
        if not match_odds or len(match_odds) < 1:
            return None

        # Find best odds for each outcome across all bookmakers
        best_home = (0.0, "")
        best_draw = (0.0, "")
        best_away = (0.0, "")

        for bookie, odds in match_odds.items():
            h = odds.get("home", 0)
            d = odds.get("draw", 0)
            a = odds.get("away", 0)

            if h > best_home[0]: best_home = (h, bookie)
            if d > best_draw[0]: best_draw = (d, bookie)
            if a > best_away[0]: best_away = (a, bookie)

        # Check for missing odds
        if best_home[0] <= 1 or best_draw[0] <= 1 or best_away[0] <= 1:
            return None

        # Calculate Implied Probability (Inverse Sum)
        implied_prob = (1.0 / best_home[0]) + (1.0 / best_draw[0]) + (1.0 / best_away[0])

        # If sum < 1.0, it's an arbitrage
        if implied_prob < 1.0:
            roi = (1.0 / implied_prob) - 1.0

            # Calculate stake distribution for 100 unit total
            # Stake_i = (Total / Implied_Prob) / Odds_i
            total_investment = 100.0
            # Target Return = Total / Implied
            target_return = total_investment / implied_prob

            s_h = target_return / best_home[0]
            s_d = target_return / best_draw[0]
            s_a = target_return / best_away[0]

            return {
                "type": "3-WAY_ARB",
                "roi": round(roi, 4),
                "profit_pct": round(roi * 100, 2),
                "implied_prob": round(implied_prob, 4),
                "opportunities": {
                    "home": {"odds": best_home[0], "bookie": best_home[1], "stake_pct": round(s_h, 2)},
                    "draw": {"odds": best_draw[0], "bookie": best_draw[1], "stake_pct": round(s_d, 2)},
                    "away": {"odds": best_away[0], "bookie": best_away[1], "stake_pct": round(s_a, 2)}
                }
            }

        return None

    def scan_2way(self, match_odds: Dict[str, Dict[str, float]], market_type: str = "dnb") -> Optional[Dict[str, Any]]:
        """
        Scans for 2-way arbitrage (e.g. DNB Home vs DNB Away, or Over/Under).

        Args:
            match_odds: dict of bookies.
            market_type: key prefix, e.g. "dnb" implies looking for "home_dnb" and "away_dnb" keys.
                         or "ou_2.5" implies "over_2.5" and "under_2.5".
        """
        if not match_odds: return None

        # Mapping logic
        if market_type == "dnb":
            keys = ("home_dnb", "away_dnb")
        elif "ou" in market_type:
            # e.g. market_type="ou_2.5"
            keys = (f"over_{market_type.split('_')[1]}", f"under_{market_type.split('_')[1]}")
        else:
            return None

        best_1 = (0.0, "")
        best_2 = (0.0, "")

        for bookie, odds in match_odds.items():
            o1 = odds.get(keys[0], 0)
            o2 = odds.get(keys[1], 0)

            if o1 > best_1[0]: best_1 = (o1, bookie)
            if o2 > best_2[0]: best_2 = (o2, bookie)

        if best_1[0] <= 1 or best_2[0] <= 1:
            return None

        implied = (1.0 / best_1[0]) + (1.0 / best_2[0])

        if implied < 1.0:
            roi = (1.0 / implied) - 1.0
            return {
                "type": f"2-WAY_ARB_{market_type.upper()}",
                "roi": round(roi, 4),
                "profit_pct": round(roi * 100, 2),
                "opportunities": {
                    keys[0]: {"odds": best_1[0], "bookie": best_1[1]},
                    keys[1]: {"odds": best_2[0], "bookie": best_2[1]}
                }
            }

        return None
