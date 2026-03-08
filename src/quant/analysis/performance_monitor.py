"""
performance_monitor.py – Otonom performans geri bildirim döngüsü.
Bahis sonuçlarını kontrol eder, PnL hesaplar ve sistem sağlığını izler.
"""
from __future__ import annotations

from loguru import logger
from src.system.container import container
from src.core.event_bus import Event

class PerformanceMonitor:
    """Sistemin finansal ve tahmin performansını izler."""

    def __init__(self):
        self.db = container.get("db")
        self.bus = container.get("event_bus")
        self.regime_kelly = container.get("regime_kelly")
        logger.debug("PerformanceMonitor başlatıldı.")

    async def update_results(self):
        """Bekleyen bahisleri kontrol eder ve sonuçlandırır."""
        pending_bets = self.db.get_pending_bets()
        if pending_bets.is_empty():
            return {}

        # İlgili maçları çek
        # match_ids = pending_bets["match_id"].unique().to_list()

        updates = 0
        pnl_change = 0.0

        for row in pending_bets.iter_rows(named=True):
            bet_id = row["bet_id"]
            mid = row["match_id"]
            selection = str(row["selection"]).lower()
            stake = row["stake"]
            odds = row["odds"]
            is_paper = row.get("is_paper", True)

            # Match verisini çek
            match_data = self.db.get_match(mid)
            if match_data.is_empty():
                continue

            m = match_data.row(0, named=True)
            status = m.get("status", "upcoming")

            # Eğer maç bitmişse
            if status in ["finished", "FT", "AET", "PEN"]:
                h_score = m.get("home_score")
                a_score = m.get("away_score")

                if h_score is not None and a_score is not None:
                    h, a = int(h_score), int(a_score)
                    won = self._check_outcome(selection, h, a)

                    if won:
                        pnl = stake * (odds - 1)
                        bet_status = "won"
                    else:
                        pnl = -stake
                        bet_status = "lost"

                    # DB Güncelle
                    self.db.update_bet_result(bet_id, pnl, bet_status)

                    # RegimeKelly Güncelle (Sadece gerçek bahisler için?)
                    # Analiz için paper trade de önemli olabilir ama Kelly state'i gerçek parayı yönetir.
                    if self.regime_kelly and not is_paper:
                        self.regime_kelly.record_result(won, pnl)

                    pnl_change += pnl
                    updates += 1

                    logger.info(f"Bahis Sonuçlandı: {bet_id} | {selection} | {h}-{a} | PnL: {pnl:.2f}")

        if updates > 0:
            return await self.check_health(pnl_change)

        return {}

    def _check_outcome(self, selection: str, h: int, a: int) -> bool:
        """Seçimin kazanıp kazanmadığını belirler."""
        # Basit string eşleştirme
        if selection in ["1", "home", "ev sahibi"]:
            return h > a
        elif selection in ["x", "draw", "beraberlik"]:
            return h == a
        elif selection in ["2", "away", "deplasman"]:
            return a > h
        elif "over" in selection and "2.5" in selection:
            return (h + a) > 2.5
        elif "under" in selection and "2.5" in selection:
            return (h + a) < 2.5
        elif "btts" in selection:
            if "yes" in selection or "var" in selection:
                return h > 0 and a > 0
            elif "no" in selection or "yok" in selection:
                return h == 0 or a == 0

        return False

    async def check_health(self, recent_pnl: float) -> dict:
        """Genel performansı analiz eder ve alarm üretir."""
        # Son 100 sonuçlanmış bahis (Paper + Real)
        stats = self.db.get_settled_bets(limit=100)

        if stats.is_empty():
            return {}

        total_pnl = stats["pnl"].sum()
        win_count = (stats["status"] == "won").sum()
        total_count = len(stats)
        win_rate = win_count / total_count if total_count > 0 else 0.0

        total_stake = stats["stake"].sum()
        roi = total_pnl / total_stake if total_stake > 0 else 0.0

        logger.info(f"Perf Check: ROI={roi:.2%}, WinRate={win_rate:.2%}, RecentPnL={recent_pnl:.2f}")

        alerts = []

        # Risk Alarmları
        if roi < -0.15:
            msg = f"📉 ROI Kritik Seviyede: {roi:.2%} (Son 100 bahis)"
            alert_payload = {"type": "risk_alert", "msg": msg, "level": "critical", "roi": roi}
            alerts.append(alert_payload)
            if self.bus:
                await self.bus.emit(Event("risk_alert", alert_payload))

        elif roi < -0.05:
            msg = f"⚠️ Performans Düşüşü: ROI {roi:.2%}"
            alert_payload = {"type": "risk_warning", "msg": msg, "level": "warning", "roi": roi}
            alerts.append(alert_payload)

        elif roi > 0.20:
            msg = f"🚀 Olağanüstü Performans: ROI {roi:.2%}"
            alert_payload = {"type": "risk_relax", "msg": msg, "level": "positive", "roi": roi}
            alerts.append(alert_payload)
            if self.bus:
                await self.bus.emit(Event("risk_relax", alert_payload))

        return {
            "roi": roi,
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "alerts": alerts
        }
