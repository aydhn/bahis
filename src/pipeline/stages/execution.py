from typing import Dict, Any, List
from loguru import logger
import asyncio
from src.pipeline.core import PipelineStage
from src.system.container import container

class ExecutionStage(PipelineStage):
    """Executes final bets and sends notifications."""

    def __init__(self):
        super().__init__("execution")
        self.notifier = container.get("notifier")

        # Optional Imports
        try:
            from src.ingestion.metric_exporter import MetricExporter
            self.metrics = MetricExporter()
        except ImportError:
            self.metrics = None

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        bets = context.get("final_bets", [])

        if not bets:
            logger.info("No bets to place this cycle.")
            return {}

        logger.info(f"Placing {len(bets)} bets...")

        for bet in bets:
            # 1. Simulate Execution
            match_id = bet.get("match_id", "?")
            selection = bet.get("selection", "?")
            stake = bet.get("stake", 0.0)

            logger.success(f"BET PLACED: {match_id} -> {selection} @ {bet.get('odds')} (Stake: {stake:.2f})")

            # 2. Notify
            if self.notifier:
                await self.notifier.send(
                    f"🎰 **BAHİS ALINDI**\n"
                    f"⚽ {match_id}\n"
                    f"🎯 Seçim: {selection}\n"
                    f"💰 Stake: {stake:.2f} TL\n"
                    f"📈 Oran: {bet.get('odds')}"
                )

            # 3. Metrics
            if self.metrics:
                self.metrics.inc_counter("bets_placed_total")
                self.metrics.observe("bet_stake_amount", stake)

        return {"execution_status": "success", "bets_placed": len(bets)}
