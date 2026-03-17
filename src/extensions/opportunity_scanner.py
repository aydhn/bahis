"""
opportunity_scanner.py - Autonomous Opportunity Scanner.

Runs asynchronously in the background, checking API streams or DB for
suddenly mispriced lines (e.g., injuries announced).
"""
import asyncio
from typing import Any, Optional
from loguru import logger
from src.core.event_bus import EventBus, Event

class OpportunityScanner:
    """Autonomous Opportunity Scanner."""

    def __init__(self, bus: EventBus):
        self.bus = bus
        self.running = False
        self.emitted_matches = set()
        logger.info("OpportunityScanner initialized.")

    async def scan(self):
        """Scans for mispriced odds periodically in the background."""
        self.running = True
        logger.info("OpportunityScanner: Scan task started.")

        while self.running:
            try:
                # Mock scanning logic. In a real system, this would query an API or DB
                # for sudden drops in odds or late injury announcements.

                # We emit a mock flash_opportunity event periodically to simulate
                # finding an edge.

                await asyncio.sleep(60) # Scan every 60 seconds

                logger.debug("OpportunityScanner: Scanning DB for opportunities...")
                from src.system.container import container
                db = container.get("db")
                if db:
                    query = """
                    SELECT match_id, home_team, away_team, home_odds, away_odds
                    FROM matches
                    WHERE status = 'upcoming'
                    AND (home_odds < 1.5 OR away_odds < 1.5)
                    LIMIT 5
                    """
                    opportunities = db.query(query)

                    if opportunities is not None and hasattr(opportunities, 'is_empty') and not opportunities.is_empty():
                        for row in opportunities.iter_rows(named=True):
                            match_id = row['match_id']
                            if match_id in self.emitted_matches:
                                continue

                            logger.success(f"OpportunityScanner: Found flash opportunity for {match_id}!")
                            self.emitted_matches.add(match_id)

                            if self.bus:
                                 selection = "HOME" if row['home_odds'] < 1.5 else "AWAY"
                                 odds = row['home_odds'] if row['home_odds'] < 1.5 else row['away_odds']
                                 await self.bus.emit(Event("flash_opportunity", data={
                                     "match_id": match_id,
                                     "reason": "Sudden odds drop detected by DB scanner",
                                     "selection": selection,
                                     "odds": odds
                                 }))

            except Exception as e:
                logger.debug(f"Exception caught in OpportunityScanner: {e}")
                await asyncio.sleep(60)

    def stop(self):
        """Stops the scanning task."""
        self.running = False
