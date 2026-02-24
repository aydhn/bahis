from typing import Dict, Any, List
from loguru import logger
import asyncio
import json
from src.pipeline.core import PipelineStage
from src.system.container import container
from src.system.config import settings

class ExecutionStage(PipelineStage):
    """Executes final bets (Live or Paper) and sends notifications."""

    def __init__(self):
        super().__init__("execution")
        self.notifier = container.get("notifier")
        self.db = container.get("db")

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

        logger.info(f"Processing {len(bets)} bets...")

        for bet in bets:
            match_id = bet.get("match_id", "?")
            selection = bet.get("selection", "?")
            stake = bet.get("stake", 0.0)
            odds = bet.get("odds", 0.0)

            # Check for Paper Trading
            is_paper = bet.get("is_paper", False) or bet.get("trading_mode") == "PAPER"
            bet["is_paper"] = is_paper

            # 1. Save to DB (Central Source of Truth)
            if self.db:
                try:
                    self.db.insert_bet(bet)
                except Exception as e:
                    logger.error(f"Failed to insert bet to DB: {e}")

            # 2. Process Execution
            if is_paper:
                logger.info(f"PAPER TRADE: {match_id} -> {selection} (Stake: {stake:.2f})")
                try:
                    # Append to paper trades log (Legacy/Backup)
                    log_file = settings.DATA_DIR / "paper_trades.jsonl"
                    log_file.parent.mkdir(parents=True, exist_ok=True)
                    with open(log_file, "a") as f:
                        f.write(json.dumps(bet) + "\n")
                except Exception as e:
                    logger.error(f"Failed to save paper trade: {e}")

                # Notify as Paper Trade
                if self.notifier:
                    await self.notifier.send(
                        f"📝 **PAPER TRADE**\n"
                        f"⚽ {match_id}\n"
                        f"🎯 {selection} @ {odds}\n"
                        f"💰 {stake:.2f} TL (Virtual)"
                    )
                continue

            # --- LIVE EXECUTION ---
            logger.success(f"BET PLACED: {match_id} -> {selection} @ {odds} (Stake: {stake:.2f})")

            # Notify Live Bet
            if self.notifier:
                await self.notifier.send(
                    f"🎰 **BAHİS ALINDI**\n"
                    f"⚽ {match_id}\n"
                    f"🎯 Seçim: {selection}\n"
                    f"💰 Stake: {stake:.2f} TL\n"
                    f"📈 Oran: {odds}"
                )

            # Metrics
            if self.metrics:
                self.metrics.inc_counter("bets_placed_total")
                self.metrics.observe("bet_stake_amount", stake)

        return {"execution_status": "success", "bets_placed": len(bets)}
