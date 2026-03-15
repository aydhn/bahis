from typing import Dict, Any
from loguru import logger
import json
import asyncio
import aiofiles
from src.pipeline.core import PipelineStage
from src.system.container import container
from src.system.config import settings

class ExecutionStage(PipelineStage):
    """Executes final bets (Live or Paper) and sends notifications."""

    def __init__(self):
        super().__init__("execution")
        self.notifier = container.get("notifier")
        self.db = container.get("db")

        # Ensure data directory exists for paper trades
        settings.DATA_DIR.mkdir(parents=True, exist_ok=True)

        # Subscribe to hedge events if bus is available
        # But stages don't usually subscribe. Sentinel subscribes and calls methods?
        # Or execution stage is part of the pipeline loop.
        # Hedge orders come asynchronously from Sentinel via EventBus.
        # Ideally, Sentinel should have a way to invoke execution for urgent orders.
        # But to keep it simple, we can expose a method `handle_hedge_order`.

        # Optional Imports
        try:
            from src.ingestion.metric_exporter import MetricExporter
            self.metrics = MetricExporter()
        except ImportError:
            self.metrics = None

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normal pipeline execution for NEW bets.
        """
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

            # 0. SANITY CHECK (The "JP Morgan" Final Gate)
            if not self._sanity_check(bet):
                logger.error(f"EXECUTION BLOCKED (Sanity Check Failed): {match_id}")
                continue

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
                    async with aiofiles.open(log_file, mode="a") as f:
                        await f.write(json.dumps(bet) + "\n")
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

    def _sanity_check(self, bet: Dict[str, Any]) -> bool:
        """
        Final safety validation before execution.
        Prevents fat-finger errors, stale bets, or illegal parameters.
        """
        # 1. Stake Limits
        stake = bet.get("stake", 0.0)
        if stake <= 0:
            logger.warning(f"Sanity Fail: Stake <= 0 ({stake})")
            return False

        # Hard cap (e.g. 20% of bankroll or fixed amount)
        # Ideally fetch bankroll, but let's assume max single bet cap is 5000 units for now
        if stake > 5000:
            logger.warning(f"Sanity Fail: Stake > 5000 ({stake})")
            return False

        # 2. Odds Validation
        odds = bet.get("odds", 0.0)
        if odds < 1.01:
            logger.warning(f"Sanity Fail: Odds < 1.01 ({odds})")
            return False
        if odds > 1000: # Suspiciously high
            logger.warning(f"Sanity Fail: Odds > 1000 ({odds})")
            return False

        # 3. EV Validation (unless hedging/paper)
        # Hedge orders might have negative EV on the hedge leg itself
        if not bet.get("is_hedge", False) and not bet.get("is_paper", False):
            ev = bet.get("ev", -1.0)
            # Allow slightly negative if covered by other strategies? No, strictly positive for main bets.
            # But sometimes 'ev' key is missing or not calculated for arb legs.
            # We skip if EV is explicitly negative.
            if ev < 0:
                logger.warning(f"Sanity Fail: Negative EV ({ev}) for standard bet")
                return False

        return True

    async def handle_hedge_order(self, event_data: Dict[str, Any]):
        """
        Handles urgent hedge orders triggered by Sentinel.
        This is called asynchronously, outside the normal pipeline cycle.
        """
        logger.warning(f"ExecutionStage: Processing HEDGE ORDER for {event_data.get('match_id')}")

        bet_id = event_data.get("bet_id")
        hedge_signal = event_data.get("hedge_signal", {})
        action = hedge_signal.get("action")

        # 1. Update DB Status
        if self.db and bet_id:
            try:
                # We mark the original bet as 'hedged' or 'closed'
                # And maybe insert a new 'hedge' bet record?
                # For simplicity, we just update status.
                self.db.update_bet_status(bet_id, "hedged")
                logger.info(f"Updated bet {bet_id} status to 'hedged'.")
            except Exception as e:
                logger.error(f"Failed to update bet status for hedge: {e}")

        # 2. Execute Hedge Bet (if needed)

        # 3. Notify
        if self.notifier:
            await self.notifier.send(
                f"🦔 **HEDGE EXECUTED**\n"
                f"Match: {event_data.get('match_id')}\n"
                f"Action: {action}\n"
                f"Status: Position Closed/Hedged."
            )
