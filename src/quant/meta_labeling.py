import numpy as np
import polars as pl
from typing import Dict, Any, Optional, List
from loguru import logger
from src.memory.db_manager import DBManager

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

class MetaLabeler:
    """
    Second-Layer Model ('The Supervisor').
    Predicts the probability that the primary model is correct based on historical performance.
    Uses LightGBM to find patterns in when the model fails (e.g., High Confidence but High Odds).
    """

    def __init__(self):
        self.db = DBManager()
        self.model: Optional[lgb.Booster] = None
        self.is_ready = False
        self.heuristic_mode = (lgb is None)

        if self.heuristic_mode:
            logger.warning("LightGBM not found. MetaLabeler running in Heuristic Mode.")
        else:
            logger.info("MetaLabeler initialized with LightGBM support.")

    def train_on_db(self):
        """
        Train the Meta-Model using real bet history.
        Joins 'bets' and 'signals' to get features (Confidence, Odds) and targets (Won/Lost).
        """
        if self.heuristic_mode:
            return

        try:
            # Join bets and signals to get features + outcome
            query = """
                SELECT
                    s.confidence,
                    s.odds,
                    s.ev,
                    b.status
                FROM bets b
                JOIN signals s ON b.match_id = s.match_id AND b.selection = s.selection
                WHERE b.status IN ('won', 'lost')
            """
            df = self.db.query(query)

            if df.height < 50:
                logger.warning(f"Not enough data to train MetaLabeler (Rows: {df.height}). Using Mock.")
                self.mock_train()
                return

            # Prepare Features: [Confidence, Odds, EV]
            X = df.select(["confidence", "odds", "ev"]).to_numpy()

            # Prepare Target: 1 if won, 0 if lost
            y = np.where(df["status"] == "won", 1, 0)

            self.train(X, y)

        except Exception as e:
            logger.error(f"MetaLabeler DB training error: {e}")
            self.mock_train()

    def train(self, X: np.ndarray, y: np.ndarray):
        """
        Train the LightGBM model.
        """
        if self.heuristic_mode:
            return

        try:
            # Dataset
            train_data = lgb.Dataset(X, label=y)

            # Parameters (Conservative to prevent overfitting on small data)
            params = {
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'learning_rate': 0.05,
                'num_leaves': 31,
                'verbosity': -1,
                'min_data_in_leaf': 20
            }

            self.model = lgb.train(params, train_data, num_boost_round=100)
            self.is_ready = True
            logger.info(f"MetaLabeler trained successfully on {len(X)} samples.")
        except Exception as e:
            logger.error(f"MetaLabeler training failed: {e}")
            self.is_ready = False

    def predict_score(self, features: Dict[str, float]) -> float:
        """
        Returns a 'Quality Score' (0.0 - 1.0).
        Higher score means the primary model is more likely to be correct.
        """
        # Feature extraction matching training [confidence, odds, ev]
        model_conf = features.get("confidence", 0.5)
        odds = features.get("odds", 2.0)
        # Calculate EV if not provided
        ev = features.get("ev", (model_conf * odds) - 1.0)
        entropy = features.get("entropy", 0.5)

        # 1. LightGBM Prediction
        if self.is_ready and self.model:
            try:
                x_vec = np.array([[model_conf, odds, ev]])
                # lightgbm predict returns probability of class 1
                score = float(self.model.predict(x_vec)[0])
                return score
            except Exception as e:
                logger.error(f"MetaLabeler prediction error: {e}")

        # 2. Heuristic Fallback (The "Quant Intuition")
        # If model isn't ready, we use logic.

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

        # Mock Data: 100 samples, 3 features (Conf, Odds, EV)
        X = np.random.rand(100, 3)
        X[:, 1] = X[:, 1] * 10.0 + 1.0 # Odds 1.0 - 11.0
        X[:, 2] = (X[:, 0] * X[:, 1]) - 1.0 # Approximate EV

        # Target: correlated with EV
        y = (X[:, 2] + np.random.normal(0, 0.5, 100) > 0).astype(int)

        self.train(X, y)
