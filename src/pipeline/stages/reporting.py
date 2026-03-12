from src.extensions.ceo_dashboard import CEODashboard
from typing import Dict, Any, Optional
import asyncio
from loguru import logger
from src.pipeline.core import PipelineStage
from src.reporting.telegram_bot import TelegramBot

class ReportingStage(PipelineStage):
    """
    Raporlama Katmanı (v2.0).
    İnteraktif Telegram Botu üzerinden sinyal ve risk raporlaması.
    """

    def __init__(self, bot_instance: Optional[TelegramBot] = None):
        super().__init__("reporting")
        self.bot = bot_instance
        try:
            self.ceo_dashboard = CEODashboard()
        except Exception as e:
            logger.debug(f"Exception caught: {e}")
            self.ceo_dashboard = None

        if self.bot is None:
            try:
                self.bot = TelegramBot()
            except Exception as e:
                logger.debug(f"Exception caught: {e}")
                self.bot = None

        # Botu başlat (Arka planda polling)
        if self.bot and getattr(self.bot, 'enabled', False):
            asyncio.create_task(self.bot.start())

        # Autonomous Performance Monitor
        try:
            from src.quant.analysis.performance_monitor import PerformanceMonitor
            self.monitor = PerformanceMonitor()
        except ImportError:
            self.monitor = None
            logger.warning("PerformanceMonitor not found.")

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        # 0. Context'i Bot'a aktar (Shared Memory)
        ctx = context.get("ctx")
        if ctx:
            self.bot.set_context(ctx)

        # 0.5 Run Performance Monitor (Autonomous Feedback Loop)
        if self.monitor:
            try:
                perf_report = await self.monitor.update_results()
                if perf_report:
                    context["performance_report"] = perf_report
                    self.bot.set_performance_report(perf_report)
                    # Alerts are handled via EventBus in monitor, but let's log ROI
                    roi = perf_report.get("roi", 0.0)
                    logger.info(f"Performance Monitor ROI: {roi:.2%}")
            except Exception as e:
                logger.error(f"Performance Monitor failed: {e}")


        # 0.7 CEODashboard Enforce Strategic Vision
        god_signal = context.get("god_signal")
        if god_signal:
            self.ceo_dashboard.enforce_strategic_vision(god_signal)

        # 0.8 God Mode Report
        if context.get("send_summary", True):
            if hasattr(self.bot, 'send_godmode_report'):
                await self.bot.send_godmode_report(ctx)
            else:
                report = self.ceo_dashboard.generate_report(ctx)
                if report:
                    await self.bot.send_message(report)

        bets = context.get("final_bets", [])

        # 1. Bahis Sinyalleri
        for bet in bets:
            await self.bot.send_bet_signal(bet)

        # 2. Risk Uyarıları (Pipeline generated)
        risk_alerts = context.get("risk_alerts", [])
        for alert in risk_alerts:
            await self.bot.send_risk_alert(alert.get("type"), alert.get("msg"))

        # 3. Executive Summary (Opsiyonel)
        if context.get("send_summary", False):
             # İleride eklenebilir
             pass

        return {}
