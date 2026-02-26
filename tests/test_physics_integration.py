import pytest
import asyncio
import polars as pl
from src.pipeline.stages.physics import PhysicsStage
from src.pipeline.context import BettingContext

@pytest.mark.asyncio
async def test_physics_stage_integration():
    # 1. Setup Mock Context
    # Matches DataFrame
    matches = pl.DataFrame([
        {
            "match_id": "test_match_1",
            "home_team": "Team A",
            "away_team": "Team B",
            "home_odds": 2.0,
            "draw_odds": 3.0,
            "away_odds": 3.5,
            "kickoff": "2023-10-27 20:00:00"
        },
        {
            "match_id": "test_match_2",
            "home_team": "Team C",
            "away_team": "Team D",
            "home_odds": 1.5,
            "draw_odds": 4.0,
            "away_odds": 5.0,
            "kickoff": "2023-10-27 21:00:00"
        }
    ])

    # Features DataFrame (Optional, physics stage should handle missing features)
    features = pl.DataFrame([
        {
            "match_id": "test_match_1",
            "home_xg": 1.5,
            "away_xg": 1.2,
            "home_win_rate": 0.5,
            "away_win_rate": 0.3
        }
    ])

    context = {
        "matches": matches,
        "features": features,
        "cycle": 1
    }

    # 2. Initialize Stage
    stage = PhysicsStage()

    # 3. Execute
    results = await stage.execute(context)

    # 4. Assertions
    # A. Structure
    assert "chaos_reports" in results
    assert "quantum_predictions" in results
    assert "ricci_report" in results # Could be None if graph building fails or not enough nodes, but key exists in defaults
    assert "geometric_potentials" in results
    assert "particle_reports" in results

    # B. Specifics
    # Chaos: Should have report for matches
    assert "test_match_1" in results["chaos_reports"]
    assert results["chaos_reports"]["test_match_1"].match_id == "test_match_1"

    # Quantum: Should have prediction
    assert "test_match_1" in results["quantum_predictions"]
    q_pred = results["quantum_predictions"]["test_match_1"]
    assert q_pred.probabilities is not None

    # Geometric: Should have potential
    assert "test_match_1" in results["geometric_potentials"]
    geo = results["geometric_potentials"]["test_match_1"]
    assert "spatial_dominance" in geo

    # Particle: Should have report (mocked observation)
    assert "test_match_1" in results["particle_reports"]
    p_rep = results["particle_reports"]["test_match_1"]
    assert p_rep.minute > 0

    # Ricci: Might be None or Report depending on graph size.
    # With 2 matches (4 teams), we have edges (A-B) and (C-D). Disconnected.
    # GraphRicci might fail on disconnected or small graphs, but our code catches exceptions.
    # Let's check if it ran without crashing.
    print(f"Ricci Report: {results.get('ricci_report')}")

if __name__ == "__main__":
    asyncio.run(test_physics_stage_integration())
