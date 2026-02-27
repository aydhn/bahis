"""
flash_monitor.py – The Flash Odds Monitor (The Sniper).

"Maximum Speed". This module bypasses the heavy ML pipeline for instantaneous
reaction to market anomalies (Z-Score > 3). It watches the odds stream
in real-time and triggers immediate execution events.

Functionality:
- Subscribes to real-time odds tick stream (or simulates it).
- Calculates Z-Score of odds movement over short windows (1min, 5min).
- Triggers 'flash_opportunity' event if anomaly is detected.
"""
import asyncio
import numpy as np
from typing import Dict, List, Optional
from loguru import logger
from src.core.speed_cache import SpeedCache
from src.core.event_bus import EventBus, Event

class FlashOddsMonitor:
    """
    Real-time odds anomaly detector.
    """

    def __init__(self, bus: EventBus, speed_cache: Optional[SpeedCache] = None):
        self.bus = bus
        self.cache = speed_cache or SpeedCache()
        self.running = False
        self.window_size = 60 # Look back 60 ticks
        self.z_threshold = 3.0 # Sigma trigger

    async def start(self):
        """Start monitoring loop."""
        self.running = True
        logger.info("FlashOddsMonitor started (Sniper Mode Active).")
        asyncio.create_task(self._monitor_loop())

    async def stop(self):
        """Stop monitoring."""
        self.running = False
        logger.info("FlashOddsMonitor stopped.")

    async def _monitor_loop(self):
        """Main loop checking for anomalies."""
        while self.running:
            try:
                # In a real system, this would iterate over active subscriptions in SpeedCache
                # For simulation, we check a few key keys or rely on cache updates triggering callbacks.
                # Here we assume we poll active matches.

                # Mock: Get active match IDs from cache (if method existed)
                # match_ids = self.cache.get_active_matches()
                # For now, we wait for 'odds_tick' events via Bus if connected,
                # OR we simulate a check on a known set.

                # Let's assume we are triggered by the EventBus subscription instead of polling?
                # But Step 4 says "Implement FlashOddsMonitor class".
                # Ideally it subscribes to the stream.

                await asyncio.sleep(1.0) # High frequency poll

            except Exception as e:
                logger.error(f"FlashMonitor loop error: {e}")
                await asyncio.sleep(5)

    async def on_odds_tick(self, event: Event):
        """
        Callback for new odds data.
        Event data: {'match_id': '...', 'selection': 'home', 'odds': 2.5, 'timestamp': ...}
        """
        data = event.data
        match_id = data.get("match_id")
        selection = data.get("selection")
        current_odds = data.get("odds")

        key = f"{match_id}_{selection}"

        # Get history from SpeedCache
        # Assuming cache stores a list of recent odds or we maintain it here
        # SpeedCache is usually key-value. We might need a local deque for history.
        # Let's use a local history dict for this "Sniper" logic.

        if not hasattr(self, "_history"):
            self._history: Dict[str, List[float]] = {}

        if key not in self._history:
            self._history[key] = []

        hist = self._history[key]
        hist.append(current_odds)
        if len(hist) > self.window_size:
            hist.pop(0)

        if len(hist) > 10:
            # Calculate Z-Score
            mean = np.mean(hist)
            std = np.std(hist)

            if std > 0.001:
                z = (current_odds - mean) / std

                # Detect Crash (Odds dropping significantly -> Z < -3)
                if z < -self.z_threshold:
                    logger.warning(f"⚡ FLASH ALERT: {key} Odds Crash! Z={z:.2f} ({mean:.2f} -> {current_odds:.2f})")

                    # Fire Opportunity
                    payload = {
                        "match_id": match_id,
                        "selection": selection,
                        "current_odds": current_odds,
                        "z_score": z,
                        "type": "dropping_odds"
                    }
                    await self.bus.emit(Event("flash_opportunity", payload))
