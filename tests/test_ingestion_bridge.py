import unittest
import asyncio
from unittest.mock import MagicMock, patch
import numpy as np
import polars as pl
from src.pipeline.stages.ingestion_bridge import IngestionBridgeStage

class TestIngestionBridgeStage(unittest.TestCase):
    @patch('src.pipeline.stages.ingestion_bridge.ZeroCopyBridge')
    def test_init_success(self, mock_bridge):
        """Test successful initialization creates ZeroCopyBridge."""
        stage = IngestionBridgeStage(shm_name="test_shm", shape=(100, 10))
        mock_bridge.assert_called_once_with(name="test_shm", shape=(100, 10), create=True)
        self.assertIsNotNone(stage.bridge)

    @patch('src.pipeline.stages.ingestion_bridge.ZeroCopyBridge')
    def test_init_failure(self, mock_bridge):
        """Test initialization failure handles exception gracefully."""
        mock_bridge.side_effect = Exception("Bridge creation failed")
        stage = IngestionBridgeStage()
        self.assertIsNone(stage.bridge)

    @patch('src.pipeline.stages.ingestion_bridge.ZeroCopyBridge')
    def test_execute_no_bridge(self, mock_bridge):
        """Test execute when bridge is None returns empty dict."""
        mock_bridge.side_effect = Exception("Fail")
        stage = IngestionBridgeStage()

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(stage.execute({"features": pl.DataFrame({"a": [1]})}))

        self.assertEqual(result, {})

    @patch('src.pipeline.stages.ingestion_bridge.ZeroCopyBridge')
    def test_execute_empty_features(self, mock_bridge):
        """Test execute with empty features returns empty dict."""
        stage = IngestionBridgeStage()

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(stage.execute({"features": pl.DataFrame()}))

        self.assertEqual(result, {})

    @patch('src.pipeline.stages.ingestion_bridge.ZeroCopyBridge')
    def test_execute_missing_features_key(self, mock_bridge):
        """Test execute when features key is missing from context."""
        stage = IngestionBridgeStage()

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(stage.execute({}))

        self.assertEqual(result, {})

    @patch('src.pipeline.stages.ingestion_bridge.ZeroCopyBridge')
    def test_execute_happy_path(self, mock_bridge):
        """Test execute extracts features, pads/truncates, and writes to SHM."""
        mock_bridge_instance = MagicMock()
        mock_bridge.return_value = mock_bridge_instance

        # We need a shape that is easy to test
        shape = (5, 12)
        stage = IngestionBridgeStage(shm_name="test_shm", shape=shape)

        # Create a sample DataFrame with some missing and some present columns
        # keys = [ "home_odds", "draw_odds", "away_odds", "over25_odds", "under25_odds", "home_xg", "away_xg", "home_xga", "away_xga", "home_win_rate", "away_win_rate", "odds_volatility" ]
        features = pl.DataFrame({
            "home_odds": [1.5, 2.0],
            "away_odds": [2.5, 3.0],
            # other columns missing to test filling logic
            "extra_col": [10.0, 20.0]
        })

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(stage.execute({"features": features}))

        # Verify bridge.write was called
        mock_bridge_instance.write.assert_called_once()

        # Check what was passed to write
        write_call_args = mock_bridge_instance.write.call_args[0]
        written_data = write_call_args[0]

        self.assertEqual(written_data.shape, shape)
        self.assertEqual(written_data.dtype, np.float32)

        # Row 0: home_odds=1.5, away_odds=2.5. draw_odds is missing, should be filled with 0.0
        # The exact order depends on `keys` in execute():
        # ["home_odds", "draw_odds", "away_odds", ...]
        # For our subset (home_odds, away_odds):
        # Home odds is 1.5, away is 2.5
        self.assertEqual(written_data[0, 0], 1.5) # home_odds
        self.assertEqual(written_data[0, 1], 2.5) # away_odds (since draw_odds is missing from df and fill_null logic only applies to columns *in* the dataframe)
        # Wait, the code says:
        # df_subset = features.select([ pl.col(k).fill_null(0.0) for k in keys if k in features.columns ])
        # So df_subset only has home_odds and away_odds in this case.
        # columns: home_odds, away_odds
        # data[0,0] = 1.5, data[0,1] = 2.5

        # Ensure returned shm_info is correct
        self.assertIn("shm_info", result)
        self.assertEqual(result["shm_info"]["name"], "test_shm")
        self.assertEqual(result["shm_info"]["shape"], shape)
        self.assertEqual(result["shm_info"]["valid_rows"], 2)

    @patch('src.pipeline.stages.ingestion_bridge.ZeroCopyBridge')
    def test_execute_bridge_write_exception(self, mock_bridge):
        """Test execute handles exceptions during bridge.write."""
        mock_bridge_instance = MagicMock()
        mock_bridge_instance.write.side_effect = Exception("Write failed")
        mock_bridge.return_value = mock_bridge_instance

        stage = IngestionBridgeStage()
        features = pl.DataFrame({"home_odds": [1.5]})

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(stage.execute({"features": features}))

        self.assertEqual(result, {})

    @patch('src.pipeline.stages.ingestion_bridge.ZeroCopyBridge')
    def test_cleanup(self, mock_bridge):
        """Test cleanup calls close and unlink on bridge."""
        mock_bridge_instance = MagicMock()
        mock_bridge.return_value = mock_bridge_instance

        stage = IngestionBridgeStage()
        stage.cleanup()

        mock_bridge_instance.close.assert_called_once()
        mock_bridge_instance.unlink.assert_called_once()

    @patch('src.pipeline.stages.ingestion_bridge.ZeroCopyBridge')
    def test_cleanup_no_bridge(self, mock_bridge):
        """Test cleanup does nothing if bridge is None."""
        mock_bridge.side_effect = Exception("Fail")
        stage = IngestionBridgeStage()

        stage.cleanup() # Should not raise

if __name__ == '__main__':
    unittest.main()
