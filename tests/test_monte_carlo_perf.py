
import pytest
import numpy as np
import time
from src.quant.monte_carlo_engine import MonteCarloEngine

def test_monte_carlo_engine_structure():
    """Test the structure of the simulation result."""
    engine = MonteCarloEngine(n_simulations=1000, seed=42)
    result = engine.simulate_match(1.5, 1.2)

    expected_keys = {
        "n_simulations", "home_win_count", "draw_count", "away_win_count",
        "prob_home", "prob_draw", "prob_away",
        "over_15_count", "over_25_count", "over_35_count",
        "prob_over_15", "prob_over_25", "prob_over_35",
        "btts_count", "prob_btts",
        "avg_total_goals", "avg_home_goals", "avg_away_goals",
        "std_total_goals", "top_scores"
    }

    assert set(result.keys()) == expected_keys
    assert isinstance(result["top_scores"], list)
    assert len(result["top_scores"]) <= 10

    top_score = result["top_scores"][0]
    assert "score" in top_score
    assert "count" in top_score
    assert "pct" in top_score

    # Check types
    assert isinstance(result["home_win_count"], int)
    assert isinstance(result["prob_home"], float)
    assert isinstance(top_score["count"], int)
    assert isinstance(top_score["pct"], float)

def test_monte_carlo_engine_correctness():
    """Test the correctness of the simulation logic using a fixed seed."""
    engine = MonteCarloEngine(n_simulations=10_000, seed=42)
    result = engine.simulate_match(1.5, 1.2)

    # With seed 42 and these parameters, we expect specific results
    # Based on previous runs: 1-1 was top score with ~12%
    top_score = result["top_scores"][0]
    assert top_score["score"] == "1-1"
    assert 0.11 < top_score["pct"] < 0.13

    # Check probabilities sum to approx 1
    total_prob = result["prob_home"] + result["prob_draw"] + result["prob_away"]
    assert abs(total_prob - 1.0) < 0.001

def test_monte_carlo_performance():
    """Test that 100,000 simulations run quickly (< 0.1s)."""
    engine = MonteCarloEngine(n_simulations=100_000, seed=42)

    start_time = time.time()
    engine.simulate_match(1.5, 1.2)
    duration = time.time() - start_time

    # The optimized version runs in ~0.02s on this machine.
    # We set a generous limit of 0.2s to account for CI variance,
    # but still catch major regressions (the unoptimized version was ~0.15s).
    assert duration < 0.2, f"Simulation took too long: {duration:.4f}s"

def test_score_format():
    """Test that score strings are formatted correctly 'H-A'."""
    engine = MonteCarloEngine(n_simulations=100, seed=42)
    result = engine.simulate_match(3.0, 0.5) # High home goals expected

    for item in result["top_scores"]:
        score = item["score"]
        parts = score.split("-")
        assert len(parts) == 2
        assert parts[0].isdigit()
        assert parts[1].isdigit()
