"""
conformal.py – Conformal Prediction (Kesin Güven Kümeleri).

Standart ML modelleri "Olasılık" (0.75) verir ama bu olasılık genellikle
kalibre değildir (aşırı özgüvenli).
Conformal Prediction, belirli bir hata oranını (örn. %5) garanti eden
bir "Tahmin Kümesi" (Prediction Set) üretir.

Örnek:
  - Model: Home=0.45, Draw=0.30, Away=0.25
  - Conformal (alpha=0.1): {Home, Draw}
  - Anlamı: "Gerçek sonuç %90 ihtimalle bu kümenin içinde."

Kullanım:
  cp = ConformalPredictor(alpha=0.1)
  cp.calibrate(calibration_data) # Geçmiş veriler (logits + true_label)
  prediction_set = cp.predict(new_logits) # {0, 1} gibi
"""
from __future__ import annotations

import numpy as np
from loguru import logger
from typing import List, Set

class ConformalPredictor:
    """
    Split Conformal Prediction for Classification.
    Guarantees that the true label is in the prediction set with probability 1-alpha.
    """

    def __init__(self, alpha: float = 0.1):
        """
        Args:
            alpha: Error tolerance (e.g., 0.1 for 90% coverage).
        """
        self.alpha = alpha
        self.q_hat = 0.0 # Conformal quantile
        self.is_calibrated = False
        self.calibration_scores: List[float] = []

    def calibrate(self, probs: np.ndarray, labels: np.ndarray):
        """
        Calibrate using a hold-out set.

        Args:
            probs: Probability estimates (N_samples, N_classes)
            labels: True class indices (N_samples,)
        """
        if len(probs) != len(labels):
            raise ValueError("Probs and labels must have same length")

        n = len(probs)
        if n < 10:
            logger.warning("Conformal calibration skipped: insufficient data (<10).")
            return

        # Score function: 1 - prob of true class (Simple non-conformity score)
        # S_i = 1 - f(x_i)[y_i]
        # Lower probability for true class -> Higher non-conformity

        scores = []
        for i in range(n):
            true_class_prob = probs[i, labels[i]]
            scores.append(1.0 - true_class_prob)

        self.calibration_scores = scores

        # Calculate q_hat: (1-alpha) quantile of scores
        # We need a score threshold q such that 1-alpha of calibration points have score <= q
        # k = ceil((n+1)(1-alpha))

        k = int(np.ceil((n + 1) * (1 - self.alpha)))
        k = min(k, n) # Clip to n

        sorted_scores = np.sort(scores)
        self.q_hat = sorted_scores[k-1] # k-th smallest value (0-indexed -> k-1)

        self.is_calibrated = True
        logger.info(f"Conformal calibrated (N={n}, alpha={self.alpha}). q_hat={self.q_hat:.4f}")

    def predict(self, probs: np.ndarray) -> List[Set[int]]:
        """
        Produce prediction sets for new samples.

        Returns:
            List of sets, where each set contains class indices (0=Home, 1=Draw, 2=Away).
        """
        if not self.is_calibrated:
            logger.warning("Predicting without calibration. Using heuristic threshold.")
            # Fallback: simple thresholding if not calibrated
            # E.g., include all classes with prob > alpha/2 ?? No, that's not conformal.
            # Let's assume a safe q_hat
            q_hat = 1.0 - 0.5 # Assume 0.5 prob threshold
        else:
            q_hat = self.q_hat

        # Prediction set includes all classes y where score(x,y) <= q_hat
        # score(x,y) = 1 - prob(y)
        # 1 - prob(y) <= q_hat  =>  prob(y) >= 1 - q_hat

        threshold = 1.0 - q_hat
        # Clip threshold to be safe (e.g. non-negative)
        threshold = max(0.0, threshold)

        results = []
        for i in range(len(probs)):
            p_vec = probs[i]
            # Select classes with p >= threshold
            pred_set = set(np.where(p_vec >= threshold)[0])

            # Ensure non-empty set (sometimes threshold is too high if calibration was weird)
            # Standard CP guarantees non-empty if alpha is reasonable, but discrete data might fail.
            if not pred_set:
                # Include the argmax at least
                pred_set.add(np.argmax(p_vec))

            results.append(pred_set)

        return results

    def check_certainty(self, pred_set: Set[int]) -> str:
        """Interpret the prediction set size."""
        size = len(pred_set)
        if size == 1:
            return "CERTAIN"
        elif size == 2:
            return "AMBIGUOUS" # e.g. {Home, Draw} -> 1X
        elif size == 3:
            return "UNCERTAIN" # {Home, Draw, Away} -> Model knows nothing
        return "UNKNOWN"
