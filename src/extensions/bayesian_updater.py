"""
bayesian_updater.py - Bayesian Odds Updater.

Adjusts the prior probability of a match outcome based on live events
(goals, red cards, xG momentum) using Bayes' theorem.
"""
from typing import Dict, Optional
from loguru import logger

class BayesianOddsUpdater:
    """Mathematical capability: Bayesian Odds Updater."""

    def __init__(self):
        logger.info("BayesianOddsUpdater initialized.")

    def update_probability(self, prior: float, likelihood_home: float, likelihood_away: float) -> float:
        """
        Adjusts the prior probability of a match outcome based on live events using Bayes' theorem.

        P(H|E) = P(E|H) * P(H) / P(E)
        where:
        - P(H) = prior (initial model probability for Home Win)
        - P(E|H) = likelihood_home (Probability of observing event E given Home Wins)
        - P(E|A) = likelihood_away (Probability of observing event E given Home does NOT win)
        - P(E) = (P(E|H) * P(H)) + (P(E|A) * (1 - P(H)))

        Example: Event E = "Home team gets a Red Card".
        """
        try:
            numerator = prior * likelihood_home
            denominator = numerator + ((1 - prior) * likelihood_away)

            if denominator == 0:
                logger.warning("Bayesian denominator is 0, returning prior.")
                return prior

            posterior = numerator / denominator

            logger.debug(f"Bayesian Update: Prior {prior:.3f} -> Posterior {posterior:.3f} | L(H)={likelihood_home:.2f}, L(A)={likelihood_away:.2f}")
            return min(max(posterior, 0.0), 1.0) # Clamp to [0, 1]

        except Exception as e:
            logger.debug(f"Exception caught in BayesianOddsUpdater: {e}")
            return prior
