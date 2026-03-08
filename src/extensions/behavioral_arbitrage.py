"""
behavioral_arbitrage.py – Behavioral Arbitrage Scanner.

Identifies irrational market pricing due to sentiment bias or recency bias.
If the implied probability exceeds the expected true probability significantly (>15%),
it signals a fade (Bearish sentiment for value betting).
"""
from dataclasses import dataclass
from loguru import logger

@dataclass
class BehavioralSignal:
    signal: str = "NEUTRAL" # BULLISH (Value), BEARISH (Trap/Overvalued), NEUTRAL
    divergence: float = 0.0
    sentiment_score: float = 0.0

class BehavioralArbitrage:
    """
    Advanced market behavior analyzer based on behavioral finance principles.
    Scans for sentiment divergence and irrational exuberance.
    """

    def __init__(self):
        logger.info("BehavioralArbitrage Engine initialized. Scanning for irrational pricing.")

    def detect_mispricing(self, match_id: str, odds: float, true_prob: float) -> BehavioralSignal:
        """
        Detects mispricing by comparing implied probability against true expected probability.

        Args:
            match_id: Unique match identifier.
            odds: The current market odds for the selection.
            true_prob: The model-calculated true probability of the selection.
        """
        if odds <= 1.0 or true_prob <= 0.0:
            return BehavioralSignal()

        implied_prob = 1.0 / odds
        divergence = implied_prob - true_prob

        signal = BehavioralSignal(divergence=divergence)

        # Overvalued threshold: if implied probability is >15% higher than our true probability,
        # it means the public is heavily betting on it blindly (recency bias/hype).
        if divergence > 0.15:
            signal.signal = "BEARISH"
            signal.sentiment_score = divergence * -1.0
            logger.debug(f"[{match_id}] Behavioral Arbitrage: Overvalued Trap detected. Divergence: {divergence:.2%}")

        # Undervalued threshold: true prob is >15% higher than implied
        # It means the market has faded a mathematically strong team (fear).
        elif divergence < -0.15:
            signal.signal = "BULLISH"
            signal.sentiment_score = abs(divergence)
            logger.debug(f"[{match_id}] Behavioral Arbitrage: Undervalued Value detected. Divergence: {divergence:.2%}")

        return signal
