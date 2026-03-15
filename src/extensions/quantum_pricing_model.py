from loguru import logger
import math
from src.core.interfaces import QuantModel
from typing import Dict, Any
from src.extensions.fast_math import njit

@njit(fastmath=True, nogil=True)
def _fast_poisson_heuristic(h_xg: float, a_xg: float) -> float:
    p_draw = 0.0
    for i in range(6):
        # Using math.exp and manual factorial since math.factorial is not easily njitted
        # We can implement a simple factorial or just compute the terms
        fact_i = 1.0
        for j in range(1, i + 1):
            fact_i *= j

        p_h_i = (math.exp(-h_xg) * (h_xg**i)) / fact_i
        p_a_i = (math.exp(-a_xg) * (a_xg**i)) / fact_i
        p_draw += p_h_i * p_a_i
    return p_draw

class QuantumPricingModel(QuantModel):
    def predict(self, context: Dict[str, Any]) -> Dict[str, Any]:
        h_xg = context.get('home_xg', 1.0)
        a_xg = context.get('away_xg', 1.0)

        # Simple Poisson heuristic for independent variables
        p_draw = _fast_poisson_heuristic(h_xg, a_xg)

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
