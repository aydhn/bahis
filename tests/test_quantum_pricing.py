from src.extensions.quantum_pricing_model import QuantumPricingModel
import math

def test_qpm():
    model = QuantumPricingModel()
    res = model.predict({"home_xg": 1.0, "away_xg": 1.0})
    # Both 1.0 -> p_draw > 0, and p_home == p_away
    assert res["prob_draw"] > 0
    assert math.isclose(res["prob_home"], res["prob_away"])
    assert math.isclose(res["prob_home"] + res["prob_draw"] + res["prob_away"], 1.0)
