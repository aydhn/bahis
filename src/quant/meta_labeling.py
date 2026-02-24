import numpy as np
from typing import Dict, Any, Optional
from loguru import logger

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

class MetaLabeler:
    """
    Second-Layer Model ('The Supervisor').
    Predicts the probability that the primary model is correct.
    Uses LightGBM if available, otherwise heuristics.
    """

    def __init__(self):
        self.model: Optional[lgb.Booster] = None
        self.is_ready = False
        self.heuristic_mode = (lgb is None)

        if self.heuristic_mode:
            logger.warning("LightGBM not found. MetaLabeler running in Heuristic Mode.")
        else:
            logger.info("MetaLabeler initialized with LightGBM support.")

    def train(self, X: np.ndarray, y: np.ndarray):
        """
        Train the Meta-Model.
        X: Feature matrix (Model Prob, Market Prob, Odds, Entropy, Variance)
        y: Binary target (1 = Bet Won, 0 = Bet Lost)
        """
        if self.heuristic_mode:
            logger.warning("Cannot train in Heuristic Mode.")
            return

        try:
            train_data = lgb.Dataset(X, label=y)
            params = {
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'verbosity': -1
            }
            self.model = lgb.train(params, train_data, num_boost_round=100)
            self.is_ready = True
            logger.info("MetaLabeler trained successfully.")
        except Exception as e:
            logger.error(f"MetaLabeler training failed: {e}")
            self.is_ready = False

    def predict_score(self, features: Dict[str, float]) -> float:
        """
        Returns a 'Quality Score' (0.0 - 1.0).
        Higher score means the primary model is more likely to be correct.
        """
        # Feature extraction
        model_conf = features.get("confidence", 0.5)
        entropy = features.get("entropy", 1.0)
        odds = features.get("odds", 2.0)

        # 1. LightGBM Prediction
        if self.is_ready and self.model:
            try:
                # Expecting specific feature order. For now, we use a simple vector.
                # [confidence, entropy, odds]
                x_vec = np.array([[model_conf, entropy, odds]])
                score = float(self.model.predict(x_vec)[0])
                return score
            except Exception as e:
                logger.error(f"MetaLabeler prediction error: {e}")

        # 2. Heuristic Fallback (The "Quant Intuition")
        # Logic: High confidence + Low Entropy = Good.
        # Logic: Extremely High Odds (>10) = suspicious unless model is very confident.

        score = model_conf

        # Penalize high entropy (uncertainty)
        if entropy > 0.8:
            score *= 0.8

        # Penalize 'Too Good To Be True' (High Odds, High Conf)
        if odds > 5.0 and model_conf > 0.7:
            score *= 0.9  # Skepticism penalty

        return max(0.0, min(1.0, score))

    def mock_train(self):
        """Train on dummy data to enable the model."""
        if self.heuristic_mode: return

        # Mock Data: 100 samples, 3 features (Conf, Entropy, Odds)
        X = np.random.rand(100, 3)
        X[:, 2] = X[:, 2] * 10.0 + 1.0 # Odds 1.0 - 11.0
        # Target: Random 0/1
        y = np.random.randint(0, 2, 100)

        self.train(X, y)
