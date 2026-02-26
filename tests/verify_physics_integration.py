
import asyncio
import sys
import os
import polars as pl
from loguru import logger

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pipeline.stages.physics import PhysicsStage
from src.pipeline.stages.risk import RiskStage
from src.reporting.telegram_bot import TelegramBot

async def verify_physics_stage():
    logger.info("Verifying PhysicsStage...")
    stage = PhysicsStage()

    # Check if engines are initialized (even if None due to missing deps, the attributes should exist)
    assert hasattr(stage, 'chaos_filter')
    assert hasattr(stage, 'quantum_brain')
    assert hasattr(stage, 'ricci_analyzer')
    assert hasattr(stage, 'fractal_analyzer')
    assert hasattr(stage, 'topology_mapper')
    assert hasattr(stage, 'path_signature')
    assert hasattr(stage, 'homology_scanner')
    assert hasattr(stage, 'gcn_graph')

    logger.info("PhysicsStage initialized successfully.")

    # Test execution with empty context (Polars DataFrame required)
    context = {"matches": pl.DataFrame(), "features": pl.DataFrame()}
    res = await stage.execute(context)
    assert "physics_reports" in res or "chaos_reports" in res # Check fallback
    logger.info("PhysicsStage execution test passed.")

async def verify_risk_stage():
    logger.info("Verifying RiskStage...")
    stage = RiskStage()
    logger.info("RiskStage initialized successfully.")

async def verify_bot():
    logger.info("Verifying TelegramBot...")
    bot = TelegramBot(token="dummy")
    assert hasattr(bot, 'handle_event')
    logger.info("TelegramBot initialized successfully.")

async def main():
    try:
        await verify_physics_stage()
        await verify_risk_stage()
        await verify_bot()
        logger.success("All integration verifications passed!")
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
