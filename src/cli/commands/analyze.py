import typer
import asyncio
from loguru import logger
from src.pipeline.core import PipelineEngine
from src.pipeline.stages.ingestion import IngestionStage
from src.pipeline.stages.features import FeatureStage
from src.pipeline.stages.inference import InferenceStage
from src.pipeline.stages.risk import RiskStage
from src.pipeline.stages.execution import ExecutionStage
from src.system.container import container
from src.system.lifecycle import lifecycle

app = typer.Typer()

async def main_async(mode: str, headless: bool):
    # Register Signals
    try:
        lifecycle.register_signal_handlers()
    except RuntimeError:
        logger.warning("Could not register signal handlers (loop likely handled by Typer/Click if async).")

    # Initialize Core Services
    try:
        container.get("db")
        container.get("cache")
        container.get("smart_cache")
    except Exception as e:
        logger.error(f"Core service init failed: {e}")
        # Proceeding might fail, but let pipeline try or exit?
        pass

    # Build Pipeline
    engine = PipelineEngine()
    engine.add_stage(IngestionStage())
    engine.add_stage(FeatureStage())
    engine.add_stage(InferenceStage())
    engine.add_stage(RiskStage())
    engine.add_stage(ExecutionStage())

    # Run
    await engine.run()

@app.command()
def run(
    mode: str = typer.Option("live", help="Mode: live | backtest"),
    headless: bool = typer.Option(True, help="Run without UI"),
):
    """Start the main analysis pipeline."""
    logger.info(f"Starting Analysis Pipeline (Mode: {mode})")
    asyncio.run(main_async(mode, headless))

if __name__ == "__main__":
    app()
