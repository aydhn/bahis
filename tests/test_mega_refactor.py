
import asyncio
import polars as pl
from loguru import logger

from src.pipeline.core import PipelineEngine, PipelineStage
from src.pipeline.stages.risk import RiskStage
from src.pipeline.context import BettingContext

class MockIngestionStage(PipelineStage):
    def __init__(self):
        super().__init__("ingestion")

    async def execute(self, context):
        logger.info("Mock Ingestion Executing")
        matches = pl.DataFrame({
            "home_team": ["TeamA"],
            "away_team": ["TeamB"],
            "home_odds": [2.0],
            "draw_odds": [3.0],
            "away_odds": [4.0]
        })
        return {"matches": matches}

class MockEnsembleStage(PipelineStage):
    def __init__(self):
        super().__init__("ensemble")

    async def execute(self, context):
        logger.info("Mock Ensemble Executing")
        return {
            "ensemble_results": [{
                "match_id": "TeamA_TeamB",
                "prob_home": 0.85, # Stronger signal
                "confidence": 0.95,
                "news_summary": "Team A has a new striker."
            }]
        }

async def run_test():
    engine = PipelineEngine()
    engine.add_stage(MockIngestionStage())
    engine.add_stage(MockEnsembleStage())
    engine.add_stage(RiskStage())

    logger.info("Starting Test Pipeline...")
    res = await engine.run_once()

    ctx = res.get("ctx")

    if ctx and ctx.final_bets:
        bet = ctx.final_bets[0]
        logger.info(f"Bet Approved! Stake: {bet['stake']}")
        print("\n=== NARRATIVE GENERATED ===\n")
        print(bet.get('narrative'))
        print("\n===========================\n")
    else:
        logger.warning("Bet still rejected.")

if __name__ == "__main__":
    asyncio.run(run_test())
