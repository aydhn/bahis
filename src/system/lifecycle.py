import asyncio
import signal
from loguru import logger

class LifecycleManager:
    """Manages application lifecycle and graceful shutdown."""

    def __init__(self):
        self.shutdown_event = asyncio.Event()
        self._tasks: list[asyncio.Task] = []

    def register_signal_handlers(self):
        """Register SIGINT and SIGTERM handlers."""
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self._handle_signal)
        logger.info("Lifecycle signals registered.")

    def _handle_signal(self):
        """Handle shutdown signal."""
        logger.warning("Shutdown signal received. Stopping services...")
        self.shutdown_event.set()

    def add_task(self, task: asyncio.Task):
        """Track a background task."""
        self._tasks.append(task)

    async def wait_for_shutdown(self):
        """Wait until shutdown event is set."""
        await self.shutdown_event.wait()
        logger.info("Stopping all background tasks...")
        for task in self._tasks:
            task.cancel()

        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        logger.success("Graceful shutdown complete.")

lifecycle = LifecycleManager()
