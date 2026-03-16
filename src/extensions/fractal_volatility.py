"""
fractal_volatility.py - Advanced Mathematical Extension for Hurst Exponent and Fractal Volatility

Calculates the Hurst Exponent for a match history time series to decide if volatility is
persistent (H > 0.5) or mean-reverting (H < 0.5).

Integration point: Used within InferenceStage to scale confidence based on fractal dimensions.
"""
from typing import Dict, Any, List
import numpy as np
from loguru import logger

class FractalVolatilityEngine:
    """Computes Hurst Exponent for advanced fractal market volatility profiling."""

    def __init__(self):
        logger.info("FractalVolatilityEngine initialized. Ready to calculate Hurst Exponent.")

    def _calculate_hurst_exponent(self, time_series: np.ndarray) -> float:
        """
        Calculate Hurst Exponent from a time series.
        """
        try:
            if len(time_series) < 10:
                return 0.5  # Neutral random walk assumption for small samples

            lags = range(2, len(time_series) // 2)
            tau = [np.sqrt(np.std(np.subtract(time_series[lag:], time_series[:-lag]))) for lag in lags]

            if len(tau) < 2:
                return 0.5

            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            hurst = poly[0] * 2.0

            # Bound Hurst between 0.0 and 1.0
            return max(0.0, min(1.0, float(hurst)))
        except Exception as e:
            logger.debug(f"Failed to calculate Hurst Exponent: {e}")
            return 0.5

    def analyze(self, match_id: str, historical_odds: List[float]) -> Dict[str, Any]:
        """
        Analyze the historical odds series to find its fractal nature.

        Returns:
            Dict containing hurst_exponent and regime ("persistent", "mean_reverting", "random_walk").
        """
        if not historical_odds or len(historical_odds) < 5:
            return {"hurst_exponent": 0.5, "fractal_regime": "random_walk", "volatility_multiplier": 1.0}

        try:
            series = np.array(historical_odds, dtype=np.float64)
            h = self._calculate_hurst_exponent(series)

            regime = "random_walk"
            multiplier = 1.0

            if h > 0.6:
                regime = "persistent"
                # Volatility will likely continue; reduce confidence/multiplier
                multiplier = 0.85
            elif h < 0.4:
                regime = "mean_reverting"
                # Odds revert to mean; increase confidence/multiplier
                multiplier = 1.15

            return {
                "hurst_exponent": round(h, 4),
                "fractal_regime": regime,
                "volatility_multiplier": round(multiplier, 2)
            }
        except Exception as e:
            logger.debug(f"FractalVolatilityEngine silent for {match_id}: {e}")
            return {"hurst_exponent": 0.5, "fractal_regime": "error", "volatility_multiplier": 1.0}
