import pytest
from typing import Dict, Any
from src.core.interfaces import QuantModel

class MockQuantModel:
    """A valid mock that implements the QuantModel protocol."""
    def predict(self, context: Dict[str, Any]) -> Dict[str, Any]:
        if "error" in context:
            raise ValueError("Test error")
        return {
            "prob_home": 0.45,
            "prob_draw": 0.25,
            "prob_away": 0.30,
            "confidence": 0.85
        }

class InvalidModel:
    """An invalid model missing the predict method."""
    def some_other_method(self):
        pass

class SignatureMismatchModel:
    """An invalid model with the wrong predict signature."""
    def predict(self, x: int, y: int) -> int:
        return x + y

def test_quant_model_protocol_valid():
    """Test that a class implementing predict() correctly matches the Protocol."""
    model = MockQuantModel()

    # Verify that the runtime checkable protocol recognizes the mock
    assert isinstance(model, QuantModel)

    # Verify the prediction behavior
    context = {"xG_home": 1.5, "xG_away": 1.1}
    result = model.predict(context)

    assert result["prob_home"] == 0.45
    assert result["confidence"] == 0.85

def test_quant_model_protocol_invalid():
    """Test that a class without predict() does not match the Protocol."""
    invalid_model = InvalidModel()

    assert not isinstance(invalid_model, QuantModel)

def test_quant_model_prediction_error():
    """Test that prediction errors propagate properly."""
    model = MockQuantModel()

    with pytest.raises(ValueError, match="Test error"):
        model.predict({"error": True})
