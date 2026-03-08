"""
sensitivity_engine.py – The Greeks (Derivative Pricing for Sports).

Calculates option-like sensitivities for betting positions using Malliavin Calculus concepts.
This treats a sports bet as a binary option.

Metrics:
  - Delta (Δ): Sensitivity of Position Value to changes in Underlying Probability.
    "If win prob changes by 1%, how much does my EV change?"
  - Gamma (Γ): Sensitivity of Delta to changes in Probability (Convexity).
    "Is my edge accelerating or decelerating?"
  - Vega (ν): Sensitivity to Volatility.
    "Does chaos help me or hurt me?"
  - Theta (Θ): Time Decay.
    "How much value do I lose as kickoff approaches?" (Not implemented yet, needs time-to-kickoff)

Usage:
    engine = SensitivityEngine()
    greeks = engine.calculate_greeks(odds=2.0, prob=0.55, volatility=0.05)
"""
from dataclasses import dataclass

@dataclass
class Greeks:
    delta: float
    gamma: float
    vega: float
    theta: float = 0.0

class SensitivityEngine:
    """
    Calculates sensitivity metrics for betting positions.
    """

    def calculate_greeks(self, odds: float, prob: float, volatility: float = 0.05) -> Greeks:
        """
        Calculate Greeks for a standard back bet.

        Value V = Stake * (Prob * Odds - 1)  (Simplified EV)
        Actually, Value of holding the ticket = Stake * Odds * Prob (if we ignore sunk cost)
        Let's treat 'Position Value' as Current Expected Return = Stake * Odds * Prob.

        Args:
            odds: Decimal odds (e.g. 2.0)
            prob: Probability of winning (0.0 - 1.0)
            volatility: Market volatility (sigma)
        """
        # Normalized Stake = 1.0

        # 1. Delta: dV/dProb
        # V = Prob * Odds
        # dV/dP = Odds
        # This means for every 1% increase in Prob, value increases by Odds * 1%
        delta = odds

        # 2. Gamma: d2V/dP2
        # For linear payoff (fixed odds), Gamma is 0.
        # BUT, if we consider that Odds themselves might change as a function of Prob (Market efficiency),
        # Odds ~= 1/Prob (Implied).
        # V = Prob * (1/Prob) = 1 (Risk-free? No).

        # Let's stick to the sensitivity of the *Edge* assuming fixed Odds (my ticket is printed).
        # Then Gamma is 0.

        # HOWEVER, let's model the "Option Value" of the bet if we can cash out.
        # Black-Scholes for Binary Option?
        # Call Option: Payoff 1 if S > K.
        # Here: Payoff 'Odds' if Outcome occurs.

        # Let's use a heuristic Gamma that represents "Information Acceleration".
        # If we are near 50/50, variance is highest.
        # Bernoulli Variance = p(1-p).
        # Gamma ~ Change in Variance?
        gamma = 0.0 # Linear instrument

        # 3. Vega: Sensitivity to Volatility
        # Standard bets are short volatility (we want stability if we have an edge).
        # Or are we? If we have an edge, we want the game to play out typically.
        # High volatility increases the chance of "Upset".
        # If we backed the Favorite (Prob > 0.5): Volatility hurts us (Vega negative).
        # If we backed the Underdog (Prob < 0.5): Volatility helps us (Vega positive).

        # Heuristic: Vega ~ (0.5 - Prob)
        # If Prob = 0.8 (Fav), Vega = -0.3 (Hates vol)
        # If Prob = 0.2 (Dog), Vega = +0.3 (Loves vol)
        vega = 0.5 - prob

        return Greeks(
            delta=float(delta),
            gamma=float(gamma),
            vega=float(vega)
        )
