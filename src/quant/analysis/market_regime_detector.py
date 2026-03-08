"""
market_regime_detector.py – Unified Market State Authority.

This module synthesizes signals from Volatility (GARCH), Chaos (Lyapunov),
and Sentiment (Odds Drift) to determine the global market regime.
It acts as the "Weather Station" for the trading system.

Regimes:
  - STABLE: Low volatility, predictable dynamics. (Green Light)
  - VOLATILE: High variance, but not chaotic. (Yellow Light - Reduced Stake)
  - CHAOTIC: Unpredictable dynamics (Butterfly Effect). (Red Light - No Bet)
  - CRASH: Extreme downside volatility. (Red Light - Liquidate)
"""
from dataclasses import dataclass
from typing import List
import numpy as np
from loguru import logger

from src.quant.risk.volatility_analyzer import VolatilityAnalyzer
from src.quant.analysis.market_sentiment import MarketSentiment
from src.quant.physics.chaos_filter import ChaosFilter

@dataclass
class RegimeMetrics:
    """Metrics defining the current regime."""
    volatility_sigma: float = 0.0
    chaos_lambda: float = 0.0
    sentiment_divergence: float = 0.0
    regime: str = "STABLE"
    confidence: float = 1.0
    description: str = ""

class MarketRegimeDetector:
    """
    Central Authority for Market State.
    """

    def __init__(self):
        self.vol_analyzer = VolatilityAnalyzer()
        self.sentiment_analyzer = MarketSentiment()
        self.chaos_filter = ChaosFilter()
        logger.info("MarketRegimeDetector initialized.")

    def detect_regime(self, match_id: str, odds_history: List[float]) -> RegimeMetrics:
        """
        Determines the market regime for a specific match context.

        Args:
            match_id: Unique identifier for the match.
            odds_history: Time series of odds (implied probabilities or raw prices).

        Returns:
            RegimeMetrics object containing the decision and underlying stats.
        """
        metrics = RegimeMetrics()

        # 1. Chaos Analysis (The Physics Layer)
        # Chaos is the most fundamental veto. If the system is chaotic, statistics fail.
        if len(odds_history) > 10:
            chaos_report = self.chaos_filter.analyze(odds_history, match_id=match_id)
            metrics.chaos_lambda = chaos_report.params.max_lyapunov

            if chaos_report.regime == "chaotic":
                metrics.regime = "CHAOTIC"
                metrics.description = f"Lyapunov Exponent {metrics.chaos_lambda:.4f} indicates chaos."
                metrics.confidence = 0.0
                return metrics # Immediate exit

        # 2. Volatility Analysis (The Statistical Layer)
        # GARCH estimation of conditional volatility.
        # We need returns for GARCH.
        if len(odds_history) > 2:
            returns = np.diff(np.log(odds_history))
            vol_report = self.vol_analyzer.analyze(returns, match_id=match_id)
            metrics.volatility_sigma = vol_report.current_volatility

            if vol_report.regime == "crisis":
                metrics.regime = "CRASH"
                metrics.description = "Volatility Crisis detected (GARCH)."
                metrics.confidence = 0.1
                return metrics
            elif vol_report.regime == "storm":
                metrics.regime = "VOLATILE"
                metrics.description = "High Volatility Storm."
                metrics.confidence = 0.5

        # 3. Sentiment Analysis (The Behavioral Layer)
        # Check for smart money divergence or herd behavior.
        sentiment = self.sentiment_analyzer.analyze_sentiment(match_id)
        metrics.sentiment_divergence = sentiment.get("sentiment_score", 0.0)

        # If sentiment is extremely bearish on our position (implied by high divergence against us)
        # For this detector, we just measure magnitude of market turbulence via sentiment
        if abs(metrics.sentiment_divergence) > 0.8:
             # Extreme shifts indicate instability
             if metrics.regime == "STABLE":
                 metrics.regime = "VOLATILE"
                 metrics.description = "Extreme Sentiment Shift detected."

        if metrics.regime == "STABLE":
            metrics.description = "Market conditions are stable."
            metrics.confidence = 1.0

        return metrics
