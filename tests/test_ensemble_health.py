
from src.quant.models.ensemble import EnsembleModel
import time

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

    # Re-initialize health manually since we injected new models after init
    ensemble.model_health = {
        name: {"last_success": time.time(), "errors": 0, "status": "HEALTHY"}
        for name in ensemble.models
    }

    # Initial State
    print("Initial Health:")
    print(ensemble.model_health["rotting_model"])

    # Simulate repeated failures
    print("\nSimulating 3 failures...")
    for i in range(3):
        res = ensemble.predict({"match_id": f"test_{i}"})
        print(f"Run {i+1}: Rotting status = {ensemble.model_health['rotting_model']['status']}")

    # Check if it's marked as ROTTED
    assert ensemble.model_health["rotting_model"]["status"] == "ROTTED"
    print("\nModel successfully marked as ROTTED.")

    # Verify it is skipped
    print("Verifying skip on next run...")
    res = ensemble.predict({"match_id": "test_4"})
    # Only stable model should contribute
    print("Prediction run complete (rotting model should be skipped).")

if __name__ == "__main__":
    test_rotting_model_logic()
