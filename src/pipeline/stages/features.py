from typing import Dict, Any
from loguru import logger
import polars as pl
from pathlib import Path
from src.pipeline.core import PipelineStage
from src.system.container import container

class FeatureStage(PipelineStage):
    """Computes features for upcoming matches."""

    def __init__(self):
        super().__init__("features")
        self.smart_cache = container.get("smart_cache")
        self.cache = container.get("cache")
        self.db = container.get("db")

        # Optional Imports
        try:
            from src.quant.analysis.time_decay import ExponentialTimeDecay
            self.time_decay = ExponentialTimeDecay(preset="moderate")
        except ImportError:
            self.time_decay = None

        try:
            from src.quant.analysis.bayesian_hierarchical import NPxGFilter
            self.npxg_filter = NPxGFilter()
        except ImportError:
            self.npxg_filter = None

        try:
            from src.core.jax_accelerator import JAXAccelerator
            self.jax_acc = JAXAccelerator()
        except ImportError:
            self.jax_acc = None

        try:
            from src.quant.physics.path_signature_engine import PathSignatureEngine
            self.path_sig = PathSignatureEngine(depth=3)
        except ImportError:
            self.path_sig = None

        # Elo System Integration
        try:
            from src.quant.models.elo_glicko_rating import EloGlickoSystem
            self.elo = EloGlickoSystem()
            self.elo_path = Path("data/elo_state.pkl")
            # Try load on init
            if not self.elo.load_state(self.elo_path):
                self.elo_loaded = False
                logger.info("Elo state not found. Will train from scratch.")
            else:
                self.elo_loaded = True
        except ImportError:
            self.elo = None
            logger.warning("EloGlickoSystem not found.")

        # Kalman Team Tracker Integration
        try:
            from src.quant.analysis.kalman_tracker import KalmanTeamTracker
            self.kalman = KalmanTeamTracker()
        except ImportError:
            self.kalman = None
            logger.warning("KalmanTeamTracker not found.")

    def _apply_elo_features(self, features: pl.DataFrame, matches: pl.DataFrame) -> pl.DataFrame:
        if not getattr(self, "elo", None):
            return features
        try:
            # If not loaded (fresh start), fetch large history. Else just recent.
            limit = 100 if self.elo_loaded else 5000
            finished = self.db.get_finished_matches(limit=limit)

            if not finished.is_empty():
                self.elo.process_batch(finished)
                self.elo_loaded = True
                self.elo.save_state(self.elo_path)

            # Predict/Score for current matches
            elo_feats = self.elo.predict_for_dataframe(matches)

            if not elo_feats.is_empty():
                # Select only relevant columns to avoid collision if any
                # We want probability and rating diffs
                cols_to_join = [c for c in elo_feats.columns if c not in ["home_team", "away_team"]]
                elo_subset = elo_feats.select(cols_to_join)

                features = features.join(elo_subset, on="match_id", how="left")
                logger.debug(f"Elo features attached. Shape: {features.shape}")
        except Exception as e:
            logger.error(f"Elo integration failed: {e}")

        return features

    def _apply_kalman_features(self, features: pl.DataFrame, matches: pl.DataFrame) -> pl.DataFrame:
        if not getattr(self, "kalman", None):
            return features
        try:
            # Check for warm-up
            if not self.kalman._filters:
                logger.info("Warming up Kalman Tracker...")
                hist_matches = self.db.get_finished_matches(limit=1000)
                if not hist_matches.is_empty():
                    hist_data = hist_matches.to_dicts()
                    self.kalman.bulk_update(hist_data)

            # Predict for current matches
            kalman_preds = []
            for row in matches.iter_rows(named=True):
                pred = self.kalman.predict_match(row["home_team"], row["away_team"])
                # Flatten/Rename for features
                kalman_preds.append({
                    "match_id": row["match_id"],
                    "kalman_home_strength": pred["home_strength"],
                    "kalman_away_strength": pred["away_strength"],
                    "kalman_home_momentum": pred["home_momentum"],
                    "kalman_away_momentum": pred["away_momentum"],
                    "kalman_prob_home": pred["prob_home"]
                })

            if kalman_preds:
                k_df = pl.DataFrame(kalman_preds)
                features = features.join(k_df, on="match_id", how="left")
                logger.debug(f"Kalman features attached. Shape: {features.shape}")

        except Exception as e:
            logger.error(f"Kalman integration failed: {e}")

        return features

    def _apply_npxg_filter(self, features: pl.DataFrame) -> pl.DataFrame:
        if not getattr(self, "npxg_filter", None):
            return features
        try:
            feature_dicts = features.to_dicts()
            feature_dicts = [self.npxg_filter.filter_features(f) for f in feature_dicts]
            return pl.DataFrame(feature_dicts)
        except Exception as e:
            logger.debug(f"[Features] npxg_filter failed: {e}")
            return features

    def _apply_path_signatures(self, features: pl.DataFrame) -> pl.DataFrame:
        if not getattr(self, "path_sig", None):
            return features
        try:
            sig_feats = self.path_sig.extract(features)
            if not sig_feats.is_empty():
                return features.join(sig_feats, on="match_id", how="left")
        except Exception as e:
            logger.error(f"PathSignature failed: {e}")
        return features

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        matches = context.get("matches", pl.DataFrame())
        if matches.is_empty():
            return {"features": pl.DataFrame()}

        cycle = context.get("cycle", 0)

        # 1. Compute Base Features
        # Using SmartCache logic from bahis.py
        def compute_fn():
             return self.db.build_feature_matrix(matches)

        # Note: In real app, cache.get_or_compute takes a key.
        # Here we mimic bahis.py: `f"features_cycle_{cycle}"`
        features = self.smart_cache.get_or_compute(
            f"features_cycle_{cycle}",
            lambda: self.cache.get_or_compute("features", compute_fn),
            persist=False,
        )

        # 1.5 Elo Features Integration (Bill Benter Logic)
        features = self._apply_elo_features(features, matches)

        # 1.6 Kalman Features Integration
        features = self._apply_kalman_features(features, matches)

        # 2. Time Decay
        if getattr(self, "time_decay", None):
            features = self.time_decay.apply_to_dataframe(features, date_col="kickoff")

        # 3. npxG Filter
        features = self._apply_npxg_filter(features)

        # 4. Path Signature Features (Geometric)
        features = self._apply_path_signatures(features)

        # 5. Acceleration
        if getattr(self, "jax_acc", None):
             features = self.jax_acc.accelerate(features)

        # 6. Volatility History (Mocked for MarketRegimeHMM / MarketGod)
        vol_history = [0.01, 0.02, 0.03, 0.04, 0.05, 0.04, 0.03, 0.02, 0.01, 0.02, 0.03]

        return {"features": features, "volatility_history": vol_history}
