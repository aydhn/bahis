import numpy as np
import polars as pl
from sklearn.neighbors import NearestNeighbors
from typing import List, Dict, Any, Optional
from loguru import logger
from src.memory.db_manager import DBManager

class SimilarityEngine:
    """
    Pattern Matching Engine using K-Nearest Neighbors.
    Finds 'Ghost Games' - historical matches that statistically resemble the current one.
    Uses 'Market Implied Similarity' (Odds Profile) to find comparables.
    """

    def __init__(self, k: int = 5):
        self.k = k
        self.model = NearestNeighbors(n_neighbors=k, algorithm='auto')
        self.history_features = None
        self.history_ids = []
        self.history_outcomes = {} # match_id -> "HOME", "DRAW", "AWAY"
        self.is_fitted = False
        self.db = DBManager()

    def load_history(self):
        """
        Load historical matches from DB and train the similarity model.
        Features used: [home_odds, draw_odds, away_odds]
        """
        try:
            # Query finished matches with valid odds
            df = self.db.query("""
                SELECT match_id, home_odds, draw_odds, away_odds, home_score, away_score
                FROM matches
                WHERE status IN ('finished', 'FT', 'AET', 'PEN')
                  AND home_odds IS NOT NULL
                  AND away_odds IS NOT NULL
                  AND home_score IS NOT NULL
            """)

            if df.is_empty():
                logger.warning("SimilarityEngine: No historical data found in DB.")
                self.mock_fit()
                return

            # Prepare Features
            # Normalize odds? No, raw odds are fine for Euclidean distance in this context
            features = df.select(["home_odds", "draw_odds", "away_odds"]).to_numpy()

            match_ids = df["match_id"].to_list()

            # Prepare Outcomes
            h_score = df["home_score"].to_numpy()
            a_score = df["away_score"].to_numpy()

            outcomes = {}
            for i, mid in enumerate(match_ids):
                if h_score[i] > a_score[i]:
                    outcomes[mid] = "HOME"
                elif a_score[i] > h_score[i]:
                    outcomes[mid] = "AWAY"
                else:
                    outcomes[mid] = "DRAW"

            self.history_outcomes = outcomes
            self.fit(features, match_ids)

        except Exception as e:
            logger.error(f"SimilarityEngine load error: {e}")
            self.mock_fit()

    def fit(self, features: np.ndarray, match_ids: List[str]):
        """
        Train the similarity engine on historical match features.
        """
        if len(features) < self.k:
            logger.warning(f"Not enough history for SimilarityEngine (need {self.k}, got {len(features)})")
            return

        self.history_features = features
        self.history_ids = match_ids
        try:
            self.model.fit(features)
            self.is_fitted = True
            logger.info(f"SimilarityEngine fitted on {len(features)} matches.")
        except Exception as e:
            logger.error(f"Failed to fit SimilarityEngine: {e}")
            self.is_fitted = False

    def find_similar(self, current_features: np.ndarray) -> Dict[str, Any]:
        """
        Find k-nearest historical matches for the given feature vector.

        Args:
            current_features: Numpy array of shape (1, 3) -> [home_odds, draw_odds, away_odds]

        Returns:
            Dict containing detailed 'ghost games' analysis.
        """
        if not self.is_fitted:
            return {}

        try:
            # Reshape if 1D
            if current_features.ndim == 1:
                current_features = current_features.reshape(1, -1)

            distances, indices = self.model.kneighbors(current_features)

            similar_matches = []
            outcomes_count = {"HOME": 0, "DRAW": 0, "AWAY": 0}

            # indices is (1, k)
            for dist, idx in zip(distances[0], indices[0]):
                match_id = self.history_ids[idx]
                outcome = self.history_outcomes.get(match_id, "UNKNOWN")

                if outcome in outcomes_count:
                    outcomes_count[outcome] += 1

                similarity_score = 1.0 / (1.0 + dist)
                similar_matches.append({
                    "match_id": match_id,
                    "distance": float(dist),
                    "similarity": float(similarity_score),
                    "outcome": outcome
                })

            # Calculate "Historical Probability"
            total = self.k
            hist_probs = {
                "prob_home": outcomes_count["HOME"] / total,
                "prob_draw": outcomes_count["DRAW"] / total,
                "prob_away": outcomes_count["AWAY"] / total
            }

            return {
                "matches": similar_matches,
                "summary": outcomes_count,
                "historical_probs": hist_probs
            }

        except Exception as e:
            logger.error(f"Error finding similar matches: {e}")
            return {}

    def mock_fit(self):
        """Mock fit for testing/initialization without DB."""
        # Create dummy data: 100 matches
        dummy_feats = np.random.rand(100, 3) * 5.0 + 1.0 # Odds 1.0-6.0
        dummy_ids = [f"history_{i}" for i in range(100)]

        outcomes = {}
        for mid in dummy_ids:
            r = np.random.rand()
            if r < 0.45: outcomes[mid] = "HOME"
            elif r < 0.75: outcomes[mid] = "AWAY"
            else: outcomes[mid] = "DRAW"

        self.history_outcomes = outcomes
        self.fit(dummy_feats, dummy_ids)
