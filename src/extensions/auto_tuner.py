"""
auto_tuner.py – Real-time Hyper-parameter Auto-Tuner.

Uses Gradient Ascent and Kelly Criterion feedback to dynamically adjust system
hyper-parameters (like the base Kelly Fraction and Risk thresholds) based on
recent rolling win rates and edge capture.
"""
from typing import Dict, List, Any
import numpy as np
from loguru import logger
from dataclasses import dataclass

@dataclass
class TunerConfig:
    learning_rate: float = 0.01
    min_kelly_frac: float = 0.05
    max_kelly_frac: float = 0.50
    momentum: float = 0.9

class AutoTuner:
    """Dynamically optimizes system parameters using online learning."""

    def __init__(self, config: TunerConfig = TunerConfig()):
        self.config = config
        self.current_kelly_frac = 0.25 # Default starting point
        self.velocity = 0.0 # For momentum in gradient ascent
        logger.info("AutoTuner initialized. Ready for dynamic hyper-parameter optimization.")

    def update(self, recent_results: List[Dict[str, Any]]) -> float:
        """
        Updates the optimal Kelly Fraction based on recent betting results.
        Maximizes the logarithmic growth rate (Kelly Criterion objective).

        Args:
            recent_results: List of dicts, e.g., [{'won': True, 'odds': 2.1, 'stake': 100}]
        Returns:
            The new recommended Kelly Fraction.
        """
        if not recent_results or len(recent_results) < 10:
            return self.current_kelly_frac

        # Calculate the gradient of the log-wealth utility function
        # U(f) = E[log(1 + f * (odds - 1))] for wins + E[log(1 - f)] for losses

        # Approximate gradient using current results
        grad = 0.0
        for res in recent_results:
            won = res.get('won', False)
            odds = res.get('odds', 2.0)

            if odds <= 1.0:
                 continue

            # Derivative of log(1 + f * b) -> b / (1 + f*b)
            # Derivative of log(1 - f) -> -1 / (1 - f)
            b = odds - 1.0
            f = self.current_kelly_frac

            if won:
                grad += b / (1.0 + f * b)
            else:
                # To prevent divide by zero or negative logs if f is close to 1
                if f >= 0.99:
                     f = 0.99
                grad -= 1.0 / (1.0 - f)

        # Average gradient over recent samples
        grad /= len(recent_results)

        # Update with momentum (Gradient Ascent)
        self.velocity = self.config.momentum * self.velocity + self.config.learning_rate * grad
        new_f = self.current_kelly_frac + self.velocity

        # Clip to safe boundaries
        self.current_kelly_frac = max(self.config.min_kelly_frac, min(new_f, self.config.max_kelly_frac))

        logger.debug(f"AutoTuner: Adjusted Kelly Fraction to {self.current_kelly_frac:.3f} (Grad: {grad:.4f})")
        return self.current_kelly_frac

    def get_current_params(self) -> Dict[str, float]:
        return {
            "kelly_fraction": self.current_kelly_frac
        }
