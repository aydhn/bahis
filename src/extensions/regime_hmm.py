"""
regime_hmm.py – Market Regime Hidden Markov Model.

Predicts the *hidden* state of the market (Stable, Volatile, Chaotic)
based on observable emissions (Returns, Volatility).

Unlike simple thresholding (GARCH), HMM models the *transition probabilities*
between states, allowing us to predict the likely NEXT state.

States:
    0: STABLE (Low Volatility, Mean Reverting)
    1: VOLATILE (High Volatility, Trending)
    2: CHAOTIC (Extreme Volatility, Random Walk)

Algorithm:
    - Baum-Welch (EM) for training parameters (A, B, pi).
    - Viterbi for decoding the most likely state sequence.
    - Forward Algorithm for predicting next state probability.
"""
import numpy as np
from loguru import logger
from dataclasses import dataclass

@dataclass
class RegimePrediction:
    current_state: int
    next_state_probs: np.ndarray # [p_stable, p_volatile, p_chaotic]
    description: str

class MarketRegimeHMM:
    """
    Simplified Gaussian HMM for 3 regimes.
    """

    def __init__(self):
        # 3 States: Stable, Volatile, Chaotic
        self.n_states = 3

        # Transition Matrix (A): P(State_t | State_t-1)
        # Initial guess: Sticky states (diagonal dominant)
        self.trans_mat = np.array([
            [0.90, 0.09, 0.01], # Stable -> Stable, Volatile, Chaos
            [0.10, 0.85, 0.05], # Volatile -> Stable, Volatile, Chaos
            [0.05, 0.20, 0.75]  # Chaos -> Stable, Volatile, Chaos
        ])

        # Emission Means (Gaussian): Expected daily return volatility (std dev)
        # Stable: 1%, Volatile: 3%, Chaos: 8%
        self.means = np.array([0.01, 0.03, 0.08])

        # Emission Variances
        self.vars = np.array([0.005, 0.01, 0.04]) # Variance of the volatility itself

        # Initial State Distribution (pi)
        self.start_prob = np.array([0.7, 0.2, 0.1])

        # Log space for parameters to use in Viterbi
        self.log_start = np.log(self.start_prob + 1e-300)
        self.log_trans = np.log(self.trans_mat + 1e-300)

        logger.info("MarketRegimeHMM initialized.")

    def _gaussian_pdf(self, x, mean, var):
        """Probability Density Function of Gaussian."""
        denom = np.sqrt(2 * np.pi * var)
        num = np.exp(-((x - mean) ** 2) / (2 * var))
        return num / denom

    def predict(self, observations: np.ndarray) -> RegimePrediction:
        """
        Decodes the current regime and predicts the next one using Viterbi/Forward.

        Args:
            observations: Array of recent volatilities (e.g. rolling std dev of returns).
                          Shape: (T,)
        """
        T = len(observations)
        if T == 0:
            return RegimePrediction(0, self.start_prob, "No Data")

        # 1. Viterbi Algorithm (Decode current state)
        # delta[t, i] = max prob of path ending in state i at time t
        delta = np.zeros((T, self.n_states))
        psi = np.zeros((T, self.n_states), dtype=int)

        # Init (Log space)
        obs_0 = observations[0]
        for i in range(self.n_states):
            log_emission = np.log(self._gaussian_pdf(obs_0, self.means[i], self.vars[i]) + 1e-300)
            delta[0, i] = self.log_start[i] + log_emission

        # Recursion (Log-space for numerical stability)
        for t in range(1, T):
            obs_t = observations[t]
            for j in range(self.n_states):
                # max_i (log_delta[t-1, i] + log_A[i, j]) + log_B[j](obs_t)
                log_emission = np.log(self._gaussian_pdf(obs_t, self.means[j], self.vars[j]) + 1e-300)
                vals = [delta[t-1, i] + self.log_trans[i, j] for i in range(self.n_states)]
                best_prev = np.argmax(vals)
                delta[t, j] = vals[best_prev] + log_emission
                psi[t, j] = best_prev

        # Termination
        current_state = int(np.argmax(delta[T-1]))

        # 2. Forecast Next State
        # P(State_t+1 | State_t) = A[current_state]
        # We need normalized probability for forecast, but we have log probs.
        log_belief = delta[T-1]
        # Softmax to get probabilities back
        belief = np.exp(log_belief - np.max(log_belief))
        belief /= np.sum(belief)

        next_probs = np.dot(belief, self.trans_mat)

        labels = ["STABLE", "VOLATILE", "CHAOTIC"]

        return RegimePrediction(
            current_state=current_state,
            next_state_probs=next_probs,
            description=f"Current: {labels[current_state]} -> Forecast: {labels[np.argmax(next_probs)]}"
        )

    def train(self, observations: np.ndarray, alpha: float = 0.1):
        """
        Online EMA training.
        Dynamically updates the means and vars based on observations without full EM overhead.
        """
        if len(observations) == 0:
            return

        # Simple online update for means and variances based on the most likely state (Viterbi path)
        T = len(observations)
        if T < 2:
            return

        # Get current state predictions to assign observations
        prediction = self.predict(observations)
        curr_state = prediction.current_state

        # Use EMA to update the mean and variance of the assigned state
        for obs in observations:
            # We assign all recent obs to the current state for a fast online update
            # (In a real EM, we'd use responsibilities, but this is a high-speed heuristic)
            diff = obs - self.means[curr_state]
            self.means[curr_state] += alpha * diff
            self.vars[curr_state] = (1 - alpha) * self.vars[curr_state] + alpha * (diff ** 2)

            # Ensure variance doesn't collapse
            self.vars[curr_state] = max(self.vars[curr_state], 1e-6)

        logger.debug(f"HMM parameters updated (EMA). New Means: {self.means}")
