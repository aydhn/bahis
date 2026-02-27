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
import random
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
        self._history: Dict[str, List[float]] = {}
        self._simulated_assets = ["match_live_001_home", "match_live_002_home", "match_live_003_away"]

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
        """
        Main loop checking for anomalies.
        In simulation mode, this generates synthetic odds ticks.
        """
        logger.info("FlashOddsMonitor: Starting Simulation Loop...")

        while self.running:
            try:
                # --- Simulation Logic ---
                # Generate random ticks for monitored assets
                for asset_key in self._simulated_assets:
                    match_id, selection = asset_key.rsplit("_", 1)

                    # 1. Get previous price or init
                    if asset_key in self._history and self._history[asset_key]:
                        prev_price = self._history[asset_key][-1]
                    else:
                        prev_price = random.uniform(1.8, 3.5)

                    # 2. Simulate Random Walk
                    # Normal drift
                    change = np.random.normal(0, 0.01)

                    # Occasional Crash (Black Swan)
                    if random.random() < 0.01: # 1% chance
                        change = -0.15 # Massive drop
                        logger.debug(f"Simulating CRASH for {asset_key}")

                    new_price = max(1.01, prev_price + change)

                    # 3. Create Event
                    tick_event = Event("odds_tick", {
                        "match_id": match_id,
                        "selection": selection,
                        "odds": new_price,
                        "timestamp": asyncio.get_event_loop().time()
                    })

                    # 4. Feed into own handler (Simulating Bus subscription)
                    await self.on_odds_tick(tick_event)

                await asyncio.sleep(0.5) # High frequency poll (2Hz)

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

        if key not in self._history:
            self._history[key] = []

        hist = self._history[key]
        hist.append(current_odds)

        # Keep window fixed
        if len(hist) > self.window_size:
            hist.pop(0)

        # Need enough data for Z-score
        if len(hist) > 10:
            # Calculate Z-Score
            mean = np.mean(hist)
            std = np.std(hist)

            if std > 0.001:
                z = (current_odds - mean) / std

                # Detect Crash (Odds dropping significantly -> Z < -3)
                # Dropping odds = Value increasing (Implied Prob up) or Market Correction
                # "Sniper" looks for rapid drops in odds (which means market thinks prob is higher)
                # If we catch it early, we might arb or value bet?
                # Actually, usually 'Dropping Odds' means "Everyone is betting on this".
                # If Z < -Threshold, it's a "Crash" (Price went down fast).

                if z < -self.z_threshold:
                    # Debounce / Suppression
                    # Ensure we didn't just fire? (Simplified: just log)

                    logger.warning(f"⚡ FLASH ALERT: {key} Odds Crash! Z={z:.2f} ({mean:.2f} -> {current_odds:.2f})")

                    # Fire Opportunity
                    payload = {
                        "match_id": match_id,
                        "selection": selection,
                        "current_odds": current_odds,
                        "z_score": float(z),
                        "type": "dropping_odds",
                        "timestamp": data.get("timestamp")
                    }
                    # Emit to global bus
                    await self.bus.emit(Event("flash_opportunity", payload))
