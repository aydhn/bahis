import unittest
import time
import shutil
import os
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

# Mocking loguru and psutil before importing SuperLogger
import sys
mock_logger = MagicMock()
mock_loguru = MagicMock()
mock_loguru.logger = mock_logger
sys.modules["loguru"] = mock_loguru

mock_psutil = MagicMock()
sys.modules["psutil"] = mock_psutil

from src.utils.super_logger import SuperLogger, DecisionContext, ModuleLogStats

class TestSuperLogger(unittest.TestCase):
    def setUp(self):
        # Reset SuperLogger singleton
        SuperLogger._instance = None
        SuperLogger._initialized = False
        self.test_log_dir = Path("test_logs_unittest")
        if self.test_log_dir.exists():
            shutil.rmtree(self.test_log_dir)
        self.sl = SuperLogger(log_dir=str(self.test_log_dir))

    def tearDown(self):
        if self.test_log_dir.exists():
            shutil.rmtree(self.test_log_dir)
        SuperLogger._instance = None
        SuperLogger._initialized = False

    def test_initialization(self):
        self.assertTrue(self.test_log_dir.exists())
        self.assertEqual(self.sl.get_log_dir(), self.test_log_dir)

    def test_get_module_logger(self):
        module_name = "test.module"
        lg = self.sl.get_module_logger(module_name)
        self.assertIn(module_name, self.sl._module_sinks)
        self.assertIn(module_name, self.sl._module_stats)

    def test_timed_success(self):
        module_name = "test.timed"
        with patch("time.perf_counter") as mock_perf:
            # Mock time: t0 = 100.0, t1 = 100.05 (50ms elapsed)
            mock_perf.side_effect = [100.0, 100.05, 100.05, 100.05]

            with self.sl.timed(module_name):
                pass

        stats = self.sl.get_module_stats(module_name)
        self.assertEqual(stats.total_entries, 1)
        self.assertAlmostEqual(stats.total_duration_ms, 50.0, places=4)
        self.assertEqual(stats.errors, 0)

    def test_timed_failure(self):
        module_name = "test.failure"
        with patch("time.perf_counter") as mock_perf:
            mock_perf.side_effect = [100.0, 100.1, 100.1] # 100ms elapsed

            with self.assertRaises(ValueError):
                with self.sl.timed(module_name):
                    raise ValueError("test error")

        stats = self.sl.get_module_stats(module_name)
        self.assertEqual(stats.total_entries, 1)
        self.assertEqual(stats.errors, 1)
        self.assertAlmostEqual(stats.total_duration_ms, 100.0, places=4)

    def test_log_decision(self):
        ctx = DecisionContext(
            module="test.decision",
            match_id="m1",
            decision="BUY",
            confidence=0.95,
            reason="High probability"
        )
        self.sl.log_decision(ctx)
        # Verify logger.bind was called (since it's a mock)
        self.assertTrue(mock_logger.bind.called)

    def test_stats_retrieval(self):
        self.sl._update_stats("mod1", 100.0, is_error=False)
        self.sl._update_stats("mod1", 200.0, is_error=True)

        stats = self.sl.get_module_stats("mod1")
        self.assertEqual(stats.total_entries, 2)
        self.assertEqual(stats.errors, 1)
        self.assertEqual(stats.avg_duration_ms, 150.0)

    def test_slowest_and_error_prone_modules(self):
        self.sl._update_stats("slow", 500.0)
        self.sl._update_stats("fast", 10.0)
        self.sl._update_stats("buggy", 50.0, is_error=True)
        self.sl._update_stats("buggy", 50.0, is_error=True)

        slowest = self.sl.get_slowest_modules(top_n=1)
        self.assertEqual(slowest[0].module, "slow")

        buggy = self.sl.get_error_prone_modules(top_n=1)
        self.assertEqual(buggy[0].module, "buggy")
        self.assertEqual(buggy[0].errors, 2)

if __name__ == "__main__":
    unittest.main()
