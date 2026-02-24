import numpy as np
from sklearn.neighbors import NearestNeighbors
from typing import List, Dict, Any, Optional
from loguru import logger

class SimilarityEngine:
    """
    Pattern Matching Engine using K-Nearest Neighbors.
    Finds 'Ghost Games' - historical matches that statistically resemble the current one.
    """

    def __init__(self, k: int = 5):
        self.k = k
        self.model = NearestNeighbors(n_neighbors=k, algorithm='auto')
        self.history_features = None
        self.history_ids = []
        self.is_fitted = False

    def fit(self, features: np.ndarray, match_ids: List[str]):
        """
        Train the similarity engine on historical match features.

        Args:
            features: Numpy array of shape (n_samples, n_features)
            match_ids: List of match IDs corresponding to the samples
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

    def find_similar(self, current_features: np.ndarray) -> List[Dict[str, Any]]:
        """
        Find k-nearest historical matches for the given feature vector.

        Args:
            current_features: Numpy array of shape (1, n_features)

        Returns:
            List of dicts containing match_id and distance (similarity score).
        """
        if not self.is_fitted:
            return []

        try:
            distances, indices = self.model.kneighbors(current_features)
            results = []

            # distances and indices are 2D arrays (1, k)
            for dist, idx in zip(distances[0], indices[0]):
                match_id = self.history_ids[idx]
                similarity_score = 1.0 / (1.0 + dist) # Convert distance to 0-1 score
                results.append({
                    "match_id": match_id,
                    "distance": float(dist),
                    "similarity": float(similarity_score)
                })

            return results
        except Exception as e:
            logger.error(f"Error finding similar matches: {e}")
            return []

    def mock_fit(self):
        """Mock fit for testing/initialization without DB."""
        # Create dummy data: 100 matches, 5 features each
        dummy_feats = np.random.rand(100, 5)
        dummy_ids = [f"history_{i}" for i in range(100)]
        self.fit(dummy_feats, dummy_ids)
