import math
from src.core.interfaces import QuantModel
from typing import Dict, Any

class QuantumPricingModel(QuantModel):
    def predict(self, context: Dict[str, Any]) -> Dict[str, Any]:
        h_xg = context.get('home_xg', 1.0)
        a_xg = context.get('away_xg', 1.0)

        # Simple Poisson heuristic for independent variables
        # P(Draw) = sum( P(home=i)*P(away=i) ) for i=0 to 5
        p_draw = 0.0
        for i in range(6):
            p_h_i = (math.exp(-h_xg) * (h_xg**i)) / math.factorial(i)
            p_a_i = (math.exp(-a_xg) * (a_xg**i)) / math.factorial(i)
            p_draw += p_h_i * p_a_i

        # We cap probability
        p_draw = min(max(p_draw, 0.0), 0.5)

        # Distribute remaining probability proportionally to xG
        remaining_prob = 1.0 - p_draw
        total_xg = h_xg + a_xg

        if total_xg > 0:
            p_home = (h_xg / total_xg) * remaining_prob
            p_away = (a_xg / total_xg) * remaining_prob
        else:
            p_home = remaining_prob / 2.0
            p_away = remaining_prob / 2.0

        return {
            "prob_home": p_home,
            "prob_draw": p_draw,
            "prob_away": p_away,
            "confidence": 0.8
        }
