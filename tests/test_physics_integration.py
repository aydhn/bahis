import pytest
import polars as pl
from src.pipeline.stages.physics import PhysicsStage

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
    assert "physics_context" in results

    phys_ctx = results["physics_context"]
    assert "test_match_1" in phys_ctx

    match_1_ctx = phys_ctx["test_match_1"]

    # B. Specifics
    # Chaos regime should exist inside physics context
    assert "chaos_regime" in match_1_ctx

    # In this physics stage refactor, it outputs everything combined into physics_context.
    # We verify that geometric mapping exists
    assert "fisher_distance" in match_1_ctx

    # 5. Ensure the data makes sense
    assert match_1_ctx["fisher_distance"] > 0
