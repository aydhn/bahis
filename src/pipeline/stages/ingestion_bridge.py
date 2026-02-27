"""
ingestion_bridge.py – Zero-Copy Ingestion Bridge.

Transfers ingested features to shared memory for low-latency access by the Inference Stage.
This acts as the writer side of the ZeroCopyBridge.
"""
from typing import Dict, Any
import numpy as np
import polars as pl
from loguru import logger
from src.pipeline.core import PipelineStage
from src.core.zero_copy_bridge import ZeroCopyBridge

class IngestionBridgeStage(PipelineStage):
    """
    Writes processed features to shared memory.
    """

    def __init__(self, shm_name: str = "quant_features", shape: tuple = (1000, 12)):
        super().__init__("ingestion_bridge")
        self.shm_name = shm_name
        self.shape = shape
        self.bridge = None

        try:
            # Initialize as creator
            self.bridge = ZeroCopyBridge(name=self.shm_name, shape=self.shape, create=True)
            logger.info(f"ZeroCopyBridge created: {self.shm_name}")
        except Exception as e:
            logger.error(f"Failed to create ZeroCopyBridge: {e}")

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extracts numerical features and writes them to shared memory.
        """
        if not self.bridge:
            return {}

        features = context.get("features", pl.DataFrame())
        if features.is_empty():
            return {}

        try:
            # Extract numerical columns used by models
            # Must match the expected schema of the reader (InferenceStage/MTL)
            # For simplicity, we select specific columns or all numeric
            # In a real HFT setup, this schema is strict.

            # Using MultiTaskBackbone.FEATURE_KEYS as reference + padding if needed
            keys = [
                "home_odds", "draw_odds", "away_odds", "over25_odds", "under25_odds",
                "home_xg", "away_xg", "home_xga", "away_xga",
                "home_win_rate", "away_win_rate", "odds_volatility"
            ]

            # Select and fill nulls
            df_subset = features.select([
                pl.col(k).fill_null(0.0) for k in keys if k in features.columns
            ])

            if df_subset.width < len(keys):
                # Handle missing columns by padding
                # (Simplified for this stage)
                pass

            data = df_subset.to_numpy().astype(np.float32)

            # Pad or truncate to match shared memory shape
            # Fixed buffer size strategy: (N_max_matches, N_features)
            # We assume N_matches <= shape[0]

            rows = min(data.shape[0], self.shape[0])
            cols = min(data.shape[1], self.shape[1])

            # Create a buffer matching the SHM shape (initialized to 0)
            buffer_data = np.zeros(self.shape, dtype=np.float32)

            # Copy data into the buffer
            buffer_data[:rows, :cols] = data[:rows, :cols]

            # Write to SHM
            self.bridge.write(buffer_data)
            logger.debug(f"Wrote {rows}x{cols} features to shared memory.")

            # Pass SHM metadata in context for the reader
            return {"shm_info": {"name": self.shm_name, "shape": self.shape, "valid_rows": rows}}

        except Exception as e:
            logger.error(f"IngestionBridge write failed: {e}")
            return {}

    def cleanup(self):
        if self.bridge:
            self.bridge.close()
            self.bridge.unlink()
