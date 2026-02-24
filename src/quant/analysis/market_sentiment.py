import numpy as np
import polars as pl
from loguru import logger
from src.memory.db_manager import DBManager

class MarketSentiment:
    """
    Analyzes market movements (Odds History) to detect Smart Money flow.
    """

    def __init__(self):
        self.db = DBManager()

    def analyze_sentiment(self, match_id: str) -> dict:
        """
        Analyzes odds history for a match to determine market sentiment.

        Returns:
            dict: {
                "sentiment_score": float (-1.0 to 1.0),
                "direction": str ("HOME", "AWAY", "NEUTRAL"),
                "volatility": float,
                "details": str
            }
        """
        try:
            # Fetch odds history
            # Expecting columns: [odds, selection, timestamp]
            # Selection should be "1", "X", "2" or "HOME", "DRAW", "AWAY"
            history = self.db.get_odds_history(match_id)

            if history.is_empty():
                return {
                    "sentiment_score": 0.0,
                    "direction": "NEUTRAL",
                    "volatility": 0.0,
                    "details": "No odds history available."
                }

            # Filter for Home Odds ("1" or "HOME")
            home_moves = history.filter(pl.col("selection").is_in(["1", "HOME"]))

            if home_moves.height < 2:
                return {
                    "sentiment_score": 0.0,
                    "direction": "NEUTRAL",
                    "volatility": 0.0,
                    "details": "Insufficient odds updates."
                }

            # Calculate movement
            # Sort by timestamp just in case
            home_moves = home_moves.sort("timestamp")

            first_odds = home_moves["odds"].head(1)[0]
            last_odds = home_moves["odds"].tail(1)[0]

            # Percentage change
            # If odds drop (2.0 -> 1.8), it's bullish for Home.
            # Change = (2.0 - 1.8) / 2.0 = 0.10 (10% drop) -> Score +
            pct_change = (first_odds - last_odds) / first_odds

            # Volatility (Standard Deviation of odds)
            volatility = home_moves["odds"].std()

            # Sentiment Score Calculation
            # 10% drop -> 0.5 score. 20% drop -> 1.0 score.
            score = np.clip(pct_change * 5.0, -1.0, 1.0)

            direction = "NEUTRAL"
            if score > 0.1:
                direction = "HOME_BULLISH"
            elif score < -0.1:
                direction = "HOME_BEARISH" # Means odds drifted UP (people betting against Home)

            return {
                "sentiment_score": float(score),
                "direction": direction,
                "volatility": float(volatility or 0.0),
                "opening_odds": float(first_odds),
                "closing_odds": float(last_odds),
                "details": f"Odds moved from {first_odds} to {last_odds} ({pct_change*100:.1f}%)"
            }

        except Exception as e:
            logger.error(f"Market Sentiment Error ({match_id}): {e}")
            return {
                "sentiment_score": 0.0,
                "direction": "ERROR",
                "volatility": 0.0,
                "details": str(e)
            }
