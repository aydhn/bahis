from typing import Dict, Any
from loguru import logger
import polars as pl
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
            from src.quant.time_decay import ExponentialTimeDecay
            self.time_decay = ExponentialTimeDecay(preset="moderate")
        except ImportError:
            self.time_decay = None

        try:
            from src.quant.bayesian_hierarchical import NPxGFilter
            self.npxg_filter = NPxGFilter()
        except ImportError:
            self.npxg_filter = None

        try:
            from src.core.jax_accelerator import JAXAccelerator
            self.jax_acc = JAXAccelerator()
        except ImportError:
            self.jax_acc = None

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

        # 2. Time Decay
        if self.time_decay:
            features = self.time_decay.apply_to_dataframe(features, date_col="kickoff")

        # 3. npxG Filter
        if self.npxg_filter:
            try:
                feature_dicts = features.to_dicts()
                feature_dicts = [self.npxg_filter.filter_features(f) for f in feature_dicts]
                features = pl.DataFrame(feature_dicts)
            except Exception as e:
                logger.debug(f"[Features] npxg_filter failed: {e}")

        # 4. Acceleration
        if self.jax_acc:
             features = self.jax_acc.accelerate(features)

        return {"features": features}
