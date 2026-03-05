import sys
from unittest.mock import MagicMock
sys.modules['numba'] = MagicMock()
sys.modules['cv2'] = MagicMock()
sys.modules['pydantic_settings'] = MagicMock()
sys.modules['stable_baselines3'] = MagicMock()
sys.modules['gymnasium'] = MagicMock()

import time
from loguru import logger
from src.quant.models.ensemble import EnsembleModel

class MockModel:
    def __init__(self, name, should_fail=False):
        self.name = name
        self.should_fail = should_fail

    def predict(self, context):
        if self.should_fail:
            return {"error": "Simulated Failure"}
        return {"prob_home": 0.5, "prob_draw": 0.3, "prob_away": 0.2, "confidence": 0.8}

def test_rotting_model_logic():
    ensemble = EnsembleModel()

    # Inject Mock Models
    ensemble.models = {
        "stable_model": MockModel("stable"),
        "rotting_model": MockModel("rotting", should_fail=True)
    }

    ensemble.weights["stable_model"] = 0.5
    ensemble.weights["rotting_model"] = 0.5

    # Re-initialize health manually since we injected new models after init
    ensemble.model_health = {
        name: {"last_success": time.time(), "errors": 0, "status": "HEALTHY"}
        for name in ensemble.models
    }

    # Simulate repeated failures
    for i in range(3):
        res = ensemble.predict({"match_id": f"test_{i}"})

    # Check if it's marked as ROTTED
    assert ensemble.model_health["rotting_model"]["status"] == "ROTTED"

    # Verify it is skipped
    res = ensemble.predict({"match_id": "test_4"})

