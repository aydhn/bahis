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
from concurrent.futures import Future

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

if __name__ == "__main__":
    unittest.main()
