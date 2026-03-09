import sys
from unittest.mock import MagicMock

# Define a simple PlayerAttributes mock to avoid importing it and triggering heavy dependencies
from dataclasses import dataclass

@dataclass
class PlayerAttributes:
    name: str = ""
    position: str = "MID"
    team: str = ""
    speed: float = 70.0
    shooting: float = 60.0
    passing: float = 65.0
    defending: float = 55.0
    stamina: float = 80.0
    fatigue: float = 0.0
    x: float = 0.0
    y: float = 0.0

    @property
    def __dataclass_fields__(self):
        # Fallback for getattr(p, "__dataclass_fields__")
        return ["name", "position", "team", "speed", "shooting", "passing", "defending", "stamina", "fatigue", "x", "y"]

# Mock missing dependencies for DistributedCore
sys.modules["loguru"] = MagicMock()
sys.modules["ray"] = MagicMock()

import unittest
from unittest.mock import patch
from concurrent.futures import Future, ProcessPoolExecutor

# Now import the module under test, but we need to mock its imports too
with patch("src.core.distributed_core.RAY_OK", False):
    from src.core.distributed_core import DistributedCore

class TestDistributedCore(unittest.TestCase):
    def setUp(self):
        # We need to make sure RAY_OK is handled
        with patch("src.core.distributed_core.RAY_OK", False):
            self.dist = DistributedCore(num_cpus=1)

    def tearDown(self):
        self.dist.shutdown()


    def test_start_already_started(self):
        """Test start returns immediately if already started."""
        with patch("src.core.distributed_core.RAY_OK", False):
            self.dist._started = True
            result = self.dist.start()
            self.assertTrue(result)
            self.assertIsNone(self.dist._pool)

    def test_start_ray_disabled_fallback(self):
        """Test start creates ProcessPoolExecutor when RAY_OK is False."""
        with patch("src.core.distributed_core.RAY_OK", False):
            result = self.dist.start()
            self.assertTrue(result)
            self.assertTrue(self.dist._started)
            self.assertIsNotNone(self.dist._pool)
            self.assertEqual(self.dist._pool._max_workers, 1)

    def test_start_ray_not_initialized(self):
        """Test start initializes ray if RAY_OK is True and not initialized."""
        with patch("src.core.distributed_core.RAY_OK", True), \
             patch("src.core.distributed_core.ray") as mock_ray, \
             patch("src.core.distributed_core.logger") as mock_logger:

            mock_ray.is_initialized.return_value = False
            mock_ray.cluster_resources.return_value = {"CPU": 1, "GPU": 0, "memory": 1000000000}

            self.dist._num_gpus = 1

            result = self.dist.start()

            self.assertTrue(result)
            self.assertTrue(self.dist._started)
            mock_ray.init.assert_called_once_with(
                ignore_reinit_error=True,
                log_to_driver=False,
                num_cpus=1,
                num_gpus=1
            )
            mock_ray.cluster_resources.assert_called_once()
            self.assertIsNone(self.dist._pool)

    def test_start_ray_already_initialized(self):
        """Test start doesn't initialize ray if already initialized."""
        with patch("src.core.distributed_core.RAY_OK", True), \
             patch("src.core.distributed_core.ray") as mock_ray, \
             patch("src.core.distributed_core.logger") as mock_logger:

            mock_ray.is_initialized.return_value = True
            mock_ray.cluster_resources.return_value = {"CPU": 1, "GPU": 0, "memory": 1000000000}

            result = self.dist.start()

            self.assertTrue(result)
            self.assertTrue(self.dist._started)
            mock_ray.init.assert_not_called()
            mock_ray.cluster_resources.assert_called_once()
            self.assertIsNone(self.dist._pool)

    def test_start_ray_exception_fallback(self):
        """Test start falls back to ProcessPoolExecutor if ray init fails."""
        with patch("src.core.distributed_core.RAY_OK", True), \
             patch("src.core.distributed_core.ray") as mock_ray, \
             patch("src.core.distributed_core.logger") as mock_logger:

            mock_ray.is_initialized.return_value = False
            mock_ray.init.side_effect = Exception("Ray init failed")

            result = self.dist.start()

            self.assertTrue(result)
            self.assertTrue(self.dist._started)
            self.assertIsNotNone(self.dist._pool)
            self.assertEqual(self.dist._pool._max_workers, 1)
            mock_logger.warning.assert_called_once()
            self.assertIn("Ray başlatılamadı", mock_logger.warning.call_args[0][0])

    def test_submit_twin_fallback(self):
        """Test submit_twin using ProcessPoolExecutor fallback."""
        with patch("src.core.distributed_core.RAY_OK", False):
            self.dist.start()

            home_players = [PlayerAttributes(name="P1", team="Home")]
            away_players = [PlayerAttributes(name="P2", team="Away")]

            future = self.dist.submit_twin(home_players, away_players, "test_match", n_sims=10)
            self.assertIsInstance(future, Future)

    def test_submit_twin_serialization(self):
        """Verify that dataclasses are converted to dicts before submission."""
        with patch("src.core.distributed_core.RAY_OK", False):
            self.dist.start()

            p1 = PlayerAttributes(name="P1", team="Home", position="FWD", speed=90)
            home_players = [p1]
            away_players = []

            # Mock the pool's submit method
            self.dist._pool.submit = MagicMock()

            initial_tasks = self.dist._tasks_in_flight
            self.dist.submit_twin(home_players, away_players, "test_match", n_sims=10)

            # Verify tasks in flight counter
            self.assertEqual(self.dist._tasks_in_flight, initial_tasks + 1)

            # Get the arguments passed to submit
            args, kwargs = self.dist._pool.submit.call_args

            # The first argument is the function: _local_run_digital_twin
            # The second is home_dicts
            home_dicts = args[1]

            self.assertIsInstance(home_dicts[0], dict)
            self.assertEqual(home_dicts[0]["name"], "P1")
            self.assertEqual(home_dicts[0]["team"], "Home")

    def test_submit_twin_ray_path(self):
        """Verify Ray execution path in submit_twin."""
        with patch("src.core.distributed_core.RAY_OK", True), \
             patch("src.core.distributed_core.ray") as mock_ray, \
             patch("src.core.distributed_core._ray_run_digital_twin") as mock_remote_func:

            mock_ray.is_initialized.return_value = True

            dist = DistributedCore(num_cpus=1)
            dist._started = True # Simulate started

            p1 = PlayerAttributes(name="P1", team="Home")
            home_players = [p1]
            away_players = []

            dist.submit_twin(home_players, away_players, "test_match", n_sims=10)

            # Verify remote was called
            mock_remote_func.remote.assert_called_once()

            # Verify arguments: should be dicts
            args, _ = mock_remote_func.remote.call_args
            home_dicts = args[0]
            self.assertIsInstance(home_dicts[0], dict)
            self.assertEqual(home_dicts[0]["name"], "P1")

    def test_submit_nash_fallback(self):
        """Test submit_nash when Ray is not available."""
        with patch("src.core.distributed_core.RAY_OK", False):
            self.dist.start()

            initial_tasks = self.dist._tasks_in_flight
            model_probs = {"home": 0.5, "draw": 0.3, "away": 0.2}
            market_odds = {"home": 2.0, "draw": 3.0, "away": 4.0}

            result = self.dist.submit_nash(model_probs, market_odds, "test_match")

            self.assertEqual(self.dist._tasks_in_flight, initial_tasks + 1)
            self.assertIsNone(result)

    def test_submit_nash_ray_path(self):
        """Verify Ray execution path in submit_nash."""
        with patch("src.core.distributed_core.RAY_OK", True), \
             patch("src.core.distributed_core.ray") as mock_ray, \
             patch("src.core.distributed_core._ray_run_nash") as mock_remote_func:

            mock_ray.is_initialized.return_value = True

            dist = DistributedCore(num_cpus=1)
            dist._started = True

            initial_tasks = dist._tasks_in_flight
            model_probs = {"home": 0.5, "draw": 0.3, "away": 0.2}
            market_odds = {"home": 2.0, "draw": 3.0, "away": 4.0}

            expected_result = MagicMock()
            mock_remote_func.remote.return_value = expected_result

            result = dist.submit_nash(model_probs, market_odds, "test_match")

            self.assertEqual(dist._tasks_in_flight, initial_tasks + 1)
            self.assertEqual(result, expected_result)
            mock_remote_func.remote.assert_called_once_with(model_probs, market_odds, "test_match")



    def test_submit_entropy_fallback(self):
        """Test submit_entropy when Ray is not available."""
        with patch("src.core.distributed_core.RAY_OK", False):
            self.dist.start()

            initial_tasks = self.dist._tasks_in_flight
            model_probs = {"home": 0.5, "draw": 0.3, "away": 0.2}
            market_odds = {"home": 2.0, "draw": 3.0, "away": 4.0}

            result = self.dist.submit_entropy(model_probs, market_odds, "test_match")

            self.assertEqual(self.dist._tasks_in_flight, initial_tasks + 1)
            self.assertIsNone(result)

    def test_submit_entropy_ray_path(self):
        """Verify Ray execution path in submit_entropy."""
        with patch("src.core.distributed_core.RAY_OK", True), \
             patch("src.core.distributed_core.ray") as mock_ray, \
             patch("src.core.distributed_core._ray_run_entropy") as mock_remote_func:

            mock_ray.is_initialized.return_value = True

            dist = DistributedCore(num_cpus=1)
            dist._started = True

            initial_tasks = dist._tasks_in_flight
            model_probs = {"home": 0.5, "draw": 0.3, "away": 0.2}
            market_odds = {"home": 2.0, "draw": 3.0, "away": 4.0}

            expected_result = MagicMock()
            mock_remote_func.remote.return_value = expected_result

            result = dist.submit_entropy(model_probs, market_odds, "test_match")

            self.assertEqual(dist._tasks_in_flight, initial_tasks + 1)
            self.assertEqual(result, expected_result)
            mock_remote_func.remote.assert_called_once_with(model_probs, market_odds, "test_match")

    def test_shutdown_ray_success(self):
        """Verify shutdown properly calls ray.shutdown when initialized."""
        with patch("src.core.distributed_core.RAY_OK", True), \
             patch("src.core.distributed_core.ray") as mock_ray:
            mock_ray.is_initialized.return_value = True

            dist = DistributedCore(num_cpus=1)
            dist._started = True

            # mock pools
            dist._pool = MagicMock()
            dist._thread_pool = MagicMock()

            dist.shutdown()

            mock_ray.shutdown.assert_called_once()
            dist._pool.shutdown.assert_called_once_with(wait=False)
            dist._thread_pool.shutdown.assert_called_once_with(wait=False)

    def test_shutdown_ray_exception(self):
        """Verify shutdown handles exceptions from ray.shutdown gracefully."""
        with patch("src.core.distributed_core.RAY_OK", True), \
             patch("src.core.distributed_core.ray") as mock_ray:
            mock_ray.is_initialized.return_value = True
            mock_ray.shutdown.side_effect = Exception("Ray shutdown failed")

            dist = DistributedCore(num_cpus=1)
            dist._started = True

            # mock pools
            dist._pool = MagicMock()
            dist._thread_pool = MagicMock()

            # Should not raise exception
            dist.shutdown()

            mock_ray.shutdown.assert_called_once()
            dist._pool.shutdown.assert_called_once_with(wait=False)
            dist._thread_pool.shutdown.assert_called_once_with(wait=False)

    def test_shutdown_pools(self):
        """Verify shutdown calls pool shutdown when Ray is not used."""
        with patch("src.core.distributed_core.RAY_OK", False):
            dist = DistributedCore(num_cpus=1)
            dist._started = True

            # mock pools
            dist._pool = MagicMock()
            dist._thread_pool = MagicMock()

            dist.shutdown()

            dist._pool.shutdown.assert_called_once_with(wait=False)
            dist._thread_pool.shutdown.assert_called_once_with(wait=False)


    def test_submit_monte_carlo_fallback(self):
        """Test submit_monte_carlo when Ray is not available."""
        with patch("src.core.distributed_core.RAY_OK", False):
            self.dist.start()

            initial_tasks = self.dist._tasks_in_flight
            result = self.dist.submit_monte_carlo(1.5, 1.2, 1000)

            self.assertEqual(self.dist._tasks_in_flight, initial_tasks + 1)
            self.assertIsNone(result)

    def test_submit_monte_carlo_ray_path(self):
        """Verify Ray execution path in submit_monte_carlo."""
        with patch("src.core.distributed_core.RAY_OK", True), \
             patch("src.core.distributed_core.ray") as mock_ray, \
             patch("src.core.distributed_core._ray_run_monte_carlo") as mock_remote_func:

            mock_ray.is_initialized.return_value = True

            dist = DistributedCore(num_cpus=1)
            dist._started = True

            initial_tasks = dist._tasks_in_flight

            expected_result = MagicMock()
            mock_remote_func.remote.return_value = expected_result

            result = dist.submit_monte_carlo(1.5, 1.2, 1000)

            self.assertEqual(dist._tasks_in_flight, initial_tasks + 1)
            self.assertEqual(result, expected_result)
            mock_remote_func.remote.assert_called_once_with(1.5, 1.2, 1000)

if __name__ == "__main__":

    unittest.main()
