"""
fast_math.py – High Performance Math Utilities.

Accelerates critical calculations using Numba JIT compilation or optimized
NumPy vectorization.

Functions:
    - fast_kelly: Vectorized Kelly Criterion calculation.
    - fast_entropy: Shannon Entropy for probability distributions.
    - fast_implied_prob: Converts Odds -> Probability with margin removal.
"""
import numpy as np
from loguru import logger

# Try importing Numba
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    logger.warning("Numba not found. Using pure NumPy fallback.")
    # Dummy decorator
    def njit(func):
        return func


@njit
def fast_kelly(p: float, b: float, fraction: float = 1.0) -> float:
    """
    Calculates Kelly Stake % for a single bet.
    f = (p * b - 1.0) / (b - 1.0)
    where b is decimal odds.
    """
    if b <= 1.0 or p <= 0.0:
        return 0.0

    f = (p * b - 1.0) / (b - 1.0)

    if f < 0:
        return 0.0

    return f * fraction

@njit
def fast_kelly_batch(probs: np.ndarray, odds: np.ndarray, fraction: float = 1.0) -> np.ndarray:
    """
    Calculates Kelly Stake % for a batch of bets.
    f* = (bp - q) / b = (p(b+1) - 1) / b? No.
    Standard: f = (p(b) - 1) / (b - 1)
    where b is decimal odds.
    """
    n = len(probs)
    stakes = np.zeros(n)

    for i in range(n):
        p = probs[i]
        b = odds[i]

        if b <= 1.0:
            stakes[i] = 0.0
            continue

        f = (p * b - 1.0) / (b - 1.0)

        # Clip to [0, 1] and apply fraction
        if f < 0:
            stakes[i] = 0.0
        else:
            stakes[i] = f * fraction

    return stakes

@njit
def fast_entropy(probs: np.ndarray) -> float:
    """
    Calculates Shannon Entropy in bits.
    H = -Sum(p * log2(p))
    """
    entropy = 0.0
    s = 0.0

    # Normalize if needed
    for x in probs:
        s += x

    if s == 0:
        return 0.0

    for x in probs:
        if x > 0:
            p_norm = x / s
            entropy -= p_norm * np.log2(p_norm)

    return entropy

@njit
def fast_implied_prob(odds: np.ndarray) -> np.ndarray:
    """
    Converts decimal odds to implied probability (margin removed via normalization).
    Input: [Home, Draw, Away] odds
    """
    raw_probs = np.zeros(len(odds))
    inv_sum = 0.0

    for i in range(len(odds)):
        if odds[i] > 0:
            inv = 1.0 / odds[i]
            raw_probs[i] = inv
            inv_sum += inv

    if inv_sum == 0:
        return raw_probs

    # Normalize
    for i in range(len(odds)):
        raw_probs[i] /= inv_sum

    return raw_probs
