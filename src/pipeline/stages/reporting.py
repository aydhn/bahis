from typing import Dict, Any
from src.pipeline.core import PipelineStage
from src.reporting.telegram import TelegramReporter

class ReportingStage(PipelineStage):
    """
    Raporlama Katmanı.
    CEO'ya (Telegram) ve Loglara (File) rapor verir.
    """

    def __init__(self):
        super().__init__("reporting")
        self.reporter = TelegramReporter()

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        bets = context.get("final_bets", [])

        # 1. Bahis Sinyalleri
        for bet in bets:
            await self.reporter.send_bet_signal(bet)

        # 2. Risk Uyarıları (context'te varsa)
        risk_alerts = context.get("risk_alerts", [])
        for alert in risk_alerts:
            await self.reporter.send_risk_alert(alert.get("type"), alert.get("msg"))

        # 3. Günlük Rapor (Cycle sonunda mı? Şimdilik her cycle sonunda özet geçmeyelim, çok spam olur)
        # Sadece önemli olayları bildir.

        return {}
