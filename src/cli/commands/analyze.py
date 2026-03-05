import typer
import asyncio
from loguru import logger
from src.system.digital_twin import DigitalTwin
from src.pipeline.core import create_default_pipeline
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
        pass

    # Build Pipeline (Enhanced with Ensemble & Reporting)
    engine = create_default_pipeline()

    if mode == "backtest":
        logger.info("Starting Dream Mode (Backtest)")
        twin = DigitalTwin()
        report = await twin.dream(n_matches=100)
        logger.success(f"Dream Mode Report: {report}")
        return

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
