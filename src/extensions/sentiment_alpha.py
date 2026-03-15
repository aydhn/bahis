"""
sentiment_alpha.py – Social & Market Sentiment Alpha Engine.

Identifies hidden momentum shifts by comparing market sentiment divergence
and team narrative hype. A philosophical addition to behavioral arbitrage.
"""
from dataclasses import dataclass
from loguru import logger

@dataclass
class SentimentAlphaSignal:
    signal: str = "NEUTRAL" # BULLISH, BEARISH, NEUTRAL
    alpha_score: float = 0.0 # -1.0 to 1.0
    narrative: str = "Market sentiment is aligned with fundamentals."

class SentimentAlphaEngine:
    """
    Evaluates raw odds against hypothetical public sentiment parameters
    to detect 'Alpha' – a mathematical edge hidden in human cognitive bias.
    """

    def __init__(self):
        logger.info("SentimentAlphaEngine initialized. Scanning for cognitive bias alpha.")

    def evaluate_alpha(self, match_id: str, odds: float, public_bias: float = 0.5, true_prob: float = 0.33) -> SentimentAlphaSignal:
        """
        Evaluate the alpha edge.

        Args:
            match_id: The match identifier.
            odds: Current decimal odds for the selection.
            public_bias: Estimated public sentiment weight (0.0 to 1.0, 0.5 is neutral).
            true_prob: The model-calculated true probability of the selection.
        """
        if odds <= 1.0 or true_prob <= 0.0:
            return SentimentAlphaSignal()

        implied_prob = 1.0 / odds

        # Alpha is generated when public bias diverges strongly from true probability,
        # but the implied probability (bookie odds) hasn't fully adjusted, or over-adjusted.

        # E.g., True Prob = 40% (0.40). Public Bias = 80% (0.80). Implied Prob = 50% (0.50).
        # The bookie is protecting against public money, but hasn't gone all the way to 80%.
        # The selection is OVERVALUED (Bearish).

        bias_delta = public_bias - true_prob
        market_delta = implied_prob - true_prob

        # Calculate a simple Alpha Score
        alpha_score = (true_prob - implied_prob) + (true_prob - public_bias) * 0.5

        signal = SentimentAlphaSignal(alpha_score=alpha_score)

        if alpha_score > 0.10:
            signal.signal = "BULLISH"
            signal.narrative = f"Positive Alpha ({alpha_score:.2f}). Market underestimating true probability amidst negative public bias."
            logger.debug(f"[{match_id}] Sentiment Alpha: BULLISH ({alpha_score:.2f})")
        elif alpha_score < -0.10:
            signal.signal = "BEARISH"
            signal.narrative = f"Negative Alpha ({alpha_score:.2f}). Market overestimating true probability due to positive public bias."
            logger.debug(f"[{match_id}] Sentiment Alpha: BEARISH ({alpha_score:.2f})")

        return signal
