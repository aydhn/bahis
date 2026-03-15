import pytest
import math
from src.utils.telegram_scenario import ScenarioSimulator, Scenario, ScenarioResult

def test_simulate():
    # Arrange
    base_features = {
        "home_xg": 1.5,
        "away_xg": 1.2,
        "home_morale": 0.8,
        "away_morale": 0.6
    }

    # Create a dummy predict_fn for testing
    def dummy_predict(features):
        return {
            "prob_home": 0.5,
            "prob_draw": 0.3,
            "prob_away": 0.2,
            "xg_home": features.get("home_xg", 1.5) * features.get("home_xg_mult", 1.0),
            "xg_away": features.get("away_xg", 1.2) * features.get("away_xg_mult", 1.0),
            "over_25": 0.6,
            "btts": 0.55
        }

    # Custom scenarios for testing
    scenarios = [
        Scenario(
            id="test_scenario",
            label="Test Senaryo",
            emoji="🧪",
            description="Bu bir test senaryosudur.",
            adjustments={
                "home_xg_mult": 1.2,
                "away_xg_mult": 0.8
            }
        )
    ]

    sim = ScenarioSimulator(predict_fn=dummy_predict, scenarios=scenarios)
    match_id = "test_match_1"

    # Act
    result = sim.simulate(match_id, "test_scenario", base_features)

    # Assert
    assert isinstance(result, ScenarioResult)
    assert result.scenario_id == "test_scenario"
    assert result.scenario_label == "🧪 Test Senaryo"

    # Check original
    assert result.original["prob_home"] == 0.5
    assert result.original["xg_home"] == 1.5
    assert result.original["xg_away"] == 1.2

    # Check adjusted
    assert result.adjusted["prob_home"] == 0.5  # In dummy predict this didn't change
    assert math.isclose(result.adjusted["xg_home"], 1.5 * 1.2)  # Applied home_xg_mult
    assert math.isclose(result.adjusted["xg_away"], 1.2 * 0.8)  # Applied away_xg_mult

    # Check impact computation
    assert "xg_home" in result.impact
    assert result.impact["xg_home"]["original"] == 1.5
    assert math.isclose(result.impact["xg_home"]["adjusted"], 1.5 * 1.2)

    assert "xg_away" in result.impact
    assert result.impact["xg_away"]["original"] == 1.2
    assert math.isclose(result.impact["xg_away"]["adjusted"], 1.2 * 0.8)

    # Telegram text is built
    assert "SENARYO: Test Senaryo" in result.telegram_text

def test_simulate_unknown_scenario():
    sim = ScenarioSimulator()
    result = sim.simulate("match_1", "unknown_scen", {})
    assert result.scenario_id == "unknown_scen"
    assert result.scenario_label == "Bilinmeyen senaryo"
    assert "tanımlı değil" in result.explanation

def test_simulate_heuristic_fallback():
    # If no predict_fn is provided, it should use the heuristic fallback
    base_features = {
        "home_xg": 2.0,
        "away_xg": 1.0,
        "home_morale": 0.8,
        "away_morale": 0.5
    }

    scenarios = [
        Scenario(
            id="boost_home",
            label="Home Boost",
            adjustments={"home_xg_mult": 2.0}
        )
    ]

    sim = ScenarioSimulator(predict_fn=None, scenarios=scenarios)

    # Act
    result = sim.simulate("match_heuristic", "boost_home", base_features)

    assert isinstance(result, ScenarioResult)
    assert result.scenario_id == "boost_home"
    assert "xg_home" in result.impact

    # The adjusted xg_home should be 2.0 * 2.0 = 4.0
    assert result.adjusted["xg_home"] == 4.0
    assert result.original["xg_home"] == 2.0

def test_simulate_uses_cache():
    # Ensure caching mechanism works to avoid re-running prediction for original features
    call_count = 0

    def predict_with_count(features):
        nonlocal call_count
        call_count += 1
        return {"prob_home": 0.5, "prob_draw": 0.3, "prob_away": 0.2, "xg_home": features.get("home_xg", 1.0)}

    scenarios = [
        Scenario(id="s1", label="S1", adjustments={"home_xg_mult": 1.5}),
        Scenario(id="s2", label="S2", adjustments={"home_xg_mult": 0.5})
    ]

    sim = ScenarioSimulator(predict_fn=predict_with_count, scenarios=scenarios)

    # First call - predict called twice (original + adjusted)
    sim.simulate("match_cached", "s1", {"home_xg": 1.0})
    assert call_count == 2

    # Second call for SAME match but DIFFERENT scenario
    # predict called once (only adjusted, original comes from cache)
    sim.simulate("match_cached", "s2", {"home_xg": 1.0})
    assert call_count == 3

def test_simulate_impact_no_change():
    # If a scenario adjustment does not produce significant difference,
    # impact dictionary should be empty and explanation should state so.
    base_features = {"prob_home": 0.5, "prob_draw": 0.3, "prob_away": 0.2, "xg_home": 1.0, "xg_away": 1.0}

    # Predict function that returns a constant output (no impact)
    def constant_predict(features):
        return {"prob_home": 0.5, "prob_draw": 0.3, "prob_away": 0.2, "xg_home": 1.0, "xg_away": 1.0}

    scenarios = [
        Scenario(
            id="useless_scenario",
            label="Useless",
            adjustments={"some_random_key": 999}
        )
    ]

    sim = ScenarioSimulator(predict_fn=constant_predict, scenarios=scenarios)
    result = sim.simulate("match_no_change", "useless_scenario", base_features)

    assert not result.impact
    assert "önemli ölçüde değiştirmiyor" in result.explanation
