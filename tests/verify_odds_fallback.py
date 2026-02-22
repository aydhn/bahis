import asyncio
import sys
import os
from unittest.mock import MagicMock, AsyncMock
import pandas as pd

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock problematic modules BEFORE importing agents
sys.modules["src.quant.probabilistic_engine"] = MagicMock()
sys.modules["src.quant.multi_task_backbone"] = MagicMock()
sys.modules["src.quant.kan_interpreter"] = MagicMock()
sys.modules["src.quant.poisson_model"] = MagicMock()
sys.modules["src.quant.monte_carlo_sim"] = MagicMock()
sys.modules["src.quant.elo_engine"] = MagicMock()
sys.modules["src.quant.dixon_coles"] = MagicMock()
sys.modules["src.quant.gradient_boosting"] = MagicMock()
sys.modules["src.quant.glm_engine"] = MagicMock()
sys.modules["src.quant.bayesian_hierarchical"] = MagicMock()
sys.modules["src.quant.lstm_forecaster"] = MagicMock()
sys.modules["src.quant.sentiment_analyzer"] = MagicMock()
sys.modules["src.quant.graph_rag"] = MagicMock()
sys.modules["src.quant.match_twin"] = MagicMock()
sys.modules["src.quant.philosophical_engine"] = MagicMock()
sys.modules["src.utils.devils_advocate"] = MagicMock()

from src.agents.analysis_orchestrator import DataIngestionAgent
from loguru import logger

async def test_odds_fallback():
    logger.info("Starting Odds Fallback Test...")
    
    # Mocks
    mock_agg = MagicMock()
    mock_validator = MagicMock()
    mock_db = MagicMock()
    
    # Setup DataIngestionAgent
    agent = DataIngestionAgent(mock_agg, mock_validator)
    
    # 1. Simulate empty DB to force fetch
    mock_agg.db.get_upcoming_matches.side_effect = [
        pd.DataFrame(), # First call empty
        pd.DataFrame([ # Second call returns data
            {"match_id": "m1", "home_team": "A", "away_team": "B", "home_odds": 1.00, "draw_odds": None, "away_odds": 2.5}, # Bad odds
            {"match_id": "m2", "home_team": "C", "away_team": "D", "home_odds": 1.50, "draw_odds": 3.0, "away_odds": 2.5}, # Good odds
        ]) 
    ]
    
    # Mock fetch_today
    mock_agg.fetch_today = AsyncMock(return_value=True)
    
    # Mock validator to pass through data
    mock_validator.validate_batch.side_effect = lambda x: x 
    
    # Run Agent
    results = await agent.run()
    
    # Verify Results
    logger.info(f"Processed {len(results)} matches.")
    
    m1 = next(r for r in results if r["match_id"] == "m1")
    logger.info(f"Match 1 Odds: H={m1['home_odds']}, D={m1['draw_odds']}, A={m1['away_odds']}")
    
    if m1["home_odds"] > 1.01 and m1["draw_odds"] > 1.01:
        logger.success("✅ Match 1: Invalid odds were fixed!")
    else:
        logger.error("❌ Match 1: Low/Missing odds NOT fixed!")

    m2 = next(r for r in results if r["match_id"] == "m2")
    if m2["home_odds"] == 1.50 and m2["draw_odds"] == 3.0:
        logger.success("✅ Match 2: Valid odds preserved!")
    else:
        logger.error("❌ Match 2: Valid odds were changed!")

if __name__ == "__main__":
    asyncio.run(test_odds_fallback())
