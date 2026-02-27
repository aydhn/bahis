"""
synthetic_engine.py – Derivative Market Pricing Engine.

Calculates fair odds for synthetic markets (Draw No Bet, Double Chance)
derived from the fundamental 1X2 probabilities. Used to find value
when the main market is efficient but derivatives are mispriced.
"""
from typing import Dict, Any, Optional

class SyntheticEngine:
    """
    Calculates fair odds for derivative markets.
    """

    def calculate_dnb(self, home_odds: float, draw_odds: float, away_odds: float) -> Dict[str, float]:
        """
        Calculates fair Draw No Bet (DNB) odds.
        DNB removes the Draw option, refunding the stake if it occurs.

        Formula:
        P(Home_DNB) = P(Home) / (P(Home) + P(Away))
        Odds(Home_DNB) = 1 / P(Home_DNB)
        """
        if home_odds <= 1 or draw_odds <= 1 or away_odds <= 1:
            return {}

        # Implied Probabilities (ignoring vig for "fair" calculation, or including it if using market odds)
        # We assume input odds are market odds. To get true fair probabilities, we should remove vig first.
        # But for relative pricing, using market implied probs is a standard approximation.

        p_h = 1.0 / home_odds
        p_a = 1.0 / away_odds

        # Renormalize excluding draw
        total = p_h + p_a
        if total == 0: return {}

        fair_p_h_dnb = p_h / total
        fair_p_a_dnb = p_a / total

        return {
            "home_dnb": round(1.0 / fair_p_h_dnb, 3),
            "away_dnb": round(1.0 / fair_p_a_dnb, 3)
        }

    def calculate_double_chance(self, home_odds: float, draw_odds: float, away_odds: float) -> Dict[str, float]:
        """
        Calculates fair Double Chance (DC) odds.
        1X (Home or Draw), X2 (Draw or Away), 12 (Home or Away).

        Formula:
        P(1X) = P(Home) + P(Draw)
        Odds(1X) = 1 / P(1X)
        """
        if home_odds <= 1 or draw_odds <= 1 or away_odds <= 1:
            return {}

        p_h = 1.0 / home_odds
        p_d = 1.0 / draw_odds
        p_a = 1.0 / away_odds

        # Simple sum of probabilities
        # Note: Sum might exceed 1.0 due to vig. That's fine, result will reflect vig.

        p_1x = p_h + p_d
        p_x2 = p_d + p_a
        p_12 = p_h + p_a

        return {
            "1x": round(1.0 / p_1x, 3) if p_1x > 0 else 0.0,
            "x2": round(1.0 / p_x2, 3) if p_x2 > 0 else 0.0,
            "12": round(1.0 / p_12, 3) if p_12 > 0 else 0.0
        }

    def analyze_value(self,
                      market_1x2: Dict[str, float],
                      market_dnb: Dict[str, float]) -> Dict[str, Any]:
        """
        Checks if the market DNB price offers better value than the synthetic DNB price derived from 1X2.

        Args:
            market_1x2: {"home": 2.5, "draw": 3.2, "away": 2.8}
            market_dnb: {"home": 1.8, "away": 1.95} (Actual bookmaker prices)
        """
        synthetic = self.calculate_dnb(
            market_1x2.get("home", 0),
            market_1x2.get("draw", 0),
            market_1x2.get("away", 0)
        )

        if not synthetic:
            return {}

        res = {}

        # Check Home DNB
        syn_h = synthetic.get("home_dnb", 0)
        mkt_h = market_dnb.get("home", 0)
        if mkt_h > syn_h * 1.02: # 2% buffer
            res["home_dnb_value"] = {
                "market": mkt_h,
                "synthetic": syn_h,
                "edge": round((mkt_h / syn_h) - 1.0, 3)
            }

        # Check Away DNB
        syn_a = synthetic.get("away_dnb", 0)
        mkt_a = market_dnb.get("away", 0)
        if mkt_a > syn_a * 1.02:
            res["away_dnb_value"] = {
                "market": mkt_a,
                "synthetic": syn_a,
                "edge": round((mkt_a / syn_a) - 1.0, 3)
            }

        return res
