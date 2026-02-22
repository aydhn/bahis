
import asyncio
import sys
import time
from pathlib import Path
from loguru import logger

# Config
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.core.bootstrap import SystemBootstrapper

async def test_full_system_boot():
    """Boots the system, lets it run for 30 seconds, then shuts down."""
    logger.info("--- SYSTEM VERIFICATION: BOOT TEST ---")
    
    # Init
    bootstrapper = SystemBootstrapper(
        mode="test", 
        headless=True, 
        telegram_enabled=False, 
        dashboard=False
    )
    
    # Boot
    await bootstrapper.boot()
    
    # Run in background
    run_task = asyncio.create_task(bootstrapper.run_forever())
    
    logger.info("System running... Waiting 30s to observe behavior.")
    await asyncio.sleep(30)
    
    # Shutdown
    logger.info("Shutting down...")
    bootstrapper.shutdown_event.set()
    await run_task
    logger.success("--- SYSTEM VERIFICATION: SUCCESS ---")

if __name__ == "__main__":
    try:
        asyncio.run(test_full_system_boot())
    except KeyboardInterrupt:
        pass
