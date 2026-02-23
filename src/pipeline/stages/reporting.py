from typing import Dict, Any, Optional
import asyncio
from src.pipeline.core import PipelineStage
from src.reporting.telegram_bot import TelegramBot

class ReportingStage(PipelineStage):
    """
    Raporlama Katmanı (v2.0).
    İnteraktif Telegram Botu üzerinden sinyal ve risk raporlaması.
    """

    def __init__(self, bot_instance: Optional[TelegramBot] = None):
        super().__init__("reporting")
        self.bot = bot_instance or TelegramBot()

        # Botu başlat (Arka planda polling)
        # Eğer dışarıdan geldiyse (Sentinel), lifecycle dışarıda yönetilir.
        # Eğer içeride yaratıldıysa (Legacy/Test), burada başlat.
        if bot_instance is None and self.bot.enabled:
            asyncio.create_task(self.bot.start())

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        # 0. Context'i Bot'a aktar (Shared Memory)
        ctx = context.get("ctx")
        if ctx:
            self.bot.set_context(ctx)

        bets = context.get("final_bets", [])

        # 1. Bahis Sinyalleri
        for bet in bets:
            await self.bot.send_bet_signal(bet)

        # 2. Risk Uyarıları
        risk_alerts = context.get("risk_alerts", [])
        for alert in risk_alerts:
            await self.bot.send_risk_alert(alert.get("type"), alert.get("msg"))

        # 3. Executive Summary (Opsiyonel)
        if context.get("send_summary", False):
             # İleride eklenebilir
             pass

        return {}
