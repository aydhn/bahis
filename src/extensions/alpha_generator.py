import asyncio
from loguru import logger
from src.core.event_bus import EventBus, Event
from src.system.container import container

class AlphaGenerator:
    """Autonomous Alpha Signal Generator."""
    def __init__(self, bus: EventBus):
        self.bus = bus
        self.running = False

    async def start(self):
        self.running = True
        logger.info("AlphaGenerator started.")
        while self.running:
            try:
                loop = asyncio.get_running_loop()
                await asyncio.to_thread(self._check_market_anomalies, loop)
            except Exception as e:
                logger.error(f"AlphaGenerator failed during check: {e}")
            await asyncio.sleep(300)

    def _check_market_anomalies(self, loop: asyncio.AbstractEventLoop):
        """Scans the database for anomalies and emits alpha signals."""
        db = container.get("db")
        if not db:
            return

        try:
            # Detect recent high-scoring leagues or teams
            query = """
            SELECT home_team, away_team, (home_score + away_score) as total_goals
            FROM matches
            WHERE status = 'FINISHED'
            ORDER BY date DESC LIMIT 50
            """
            recent_matches = db.query(query)
            if recent_matches is not None and len(recent_matches) > 0:
                if hasattr(recent_matches, 'is_empty') and not recent_matches.is_empty():
                    avg_goals = recent_matches["total_goals"].mean()
                elif isinstance(recent_matches, list) and isinstance(recent_matches[0], dict):
                    total = sum(r.get("total_goals", 0) for r in recent_matches)
                    avg_goals = total / len(recent_matches)
                else:
                    avg_goals = 0.0
                if avg_goals > 3.5:
                    # High goal anomaly detected globally
                    if self.bus:
                        asyncio.run_coroutine_threadsafe(
                            self.bus.emit(Event("alpha_signal", {
                                "type": "Global_Over_Anomaly",
                                "confidence": 0.85,
                                "avg_goals": avg_goals
                            })),
                            loop
                        )
        except Exception as e:
            logger.error(f"AlphaGenerator DB error: {e}")

    def stop(self):
        self.running = False
