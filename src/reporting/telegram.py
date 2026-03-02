"""
telegram.py – CEO Paneli ve Bildirim Merkezi.

Bu modül, sistemin dış dünyaya (CEO'ya) rapor vermesini sağlar.
Yetenekler:
  - Anlık bahis sinyalleri (Rich formatting)
  - Risk uyarıları (Drawdown, Regime Change)
  - Günlük özet (PnL, Win Rate)
  - Hata raporlama

Teknoloji:
  - Async HTTP (httpx) ile bloklamayan gönderim
  - HTML/Markdown formatting
  - Retry mekanizması
"""
import os
from typing import Any, Dict, Optional

from loguru import logger

try:
    import httpx
except ImportError:
    httpx = None


class TelegramReporter:
    """Async Telegram Raporlama Servisi."""

    def __init__(self, token: Optional[str] = None, chat_id: Optional[str] = None):
        self.token = token or os.getenv("TELEGRAM_TOKEN")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")
        self.base_url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        self.enabled = bool(self.token and self.chat_id and httpx)

        if not self.enabled:
            logger.warning(
                "TelegramReporter devre dışı: Token, Chat ID veya httpx eksik."
            )
        else:
            logger.info("TelegramReporter aktif.")

    async def send_message(self, text: str, parse_mode: str = "HTML") -> bool:
        """Genel mesaj gönder."""
        if not self.enabled:
            return False

        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": parse_mode,
            "disable_web_page_preview": True,
        }

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(self.base_url, json=payload)
                if resp.status_code != 200:
                    logger.error(f"Telegram Hatası ({resp.status_code}): {resp.text}")
                    return False
                return True
        except Exception as e:
            logger.error(f"Telegram Bağlantı Hatası: {e}")
            return False

    async def send_bet_signal(self, bet: Dict[str, Any]) -> bool:
        """Bahis sinyali formatlayıp gönder."""
        if not self.enabled:
            return False

        # Formatlama (CEO Vizyonu: Net, Vurgulu, Profesyonel)
        match = bet.get("match_id", "Unknown Match")
        selection = bet.get("selection", "UNKNOWN")
        odds = bet.get("odds", 0.0)
        stake = bet.get("stake", 0.0)
        conf = bet.get("confidence", 0.0)
        reason = bet.get("reason", "")
        regime = bet.get("regime", "Unknown")

        # Emoji seçimi
        emoji = "🟢" if conf > 0.7 else "🟡"
        if conf > 0.85:
            emoji = "🔥"

        msg = (
            f"<b>{emoji} YENİ POZİSYON ALINDI</b>\n\n"
            f"⚽ <b>Maç:</b> {match}\n"
            f"🎯 <b>Seçim:</b> {selection}\n"
            f"📈 <b>Oran:</b> {odds:.2f}\n"
            f"💰 <b>Stake:</b> {stake:.2f} TL\n"
            f"🧠 <b>Güven:</b> %{conf*100:.1f}\n"
            f"🌪 <b>Rejim:</b> {regime}\n"
            f"📝 <b>Gerekçe:</b> {reason}\n"
            f"<i>🤖 Otonom Quant Sistemi</i>"
        )
        return await self.send_message(msg)

    async def send_risk_alert(self, alert_type: str, details: str) -> bool:
        """Kritik risk uyarısı."""
        if not self.enabled:
            return False

        msg = (
            f"<b>🚨 RİSK UYARISI: {alert_type.upper()}</b>\n\n"
            f"{details}\n\n"
            f"<i>Sistem koruma protokolleri devrede.</i>"
        )
        return await self.send_message(msg)

    async def send_daily_report(self, stats: Dict[str, Any]) -> bool:
        """Günlük PnL ve performans özeti."""
        if not self.enabled:
            return False

        pnl = stats.get("daily_pnl", 0.0)
        roi = stats.get("daily_roi", 0.0)
        wins = stats.get("wins", 0)
        losses = stats.get("losses", 0)

        emoji = "📈" if pnl >= 0 else "📉"

        msg = (
            f"<b>{emoji} GÜNLÜK KAPANIŞ RAPORU</b>\n\n"
            f"💵 <b>PnL:</b> {pnl:+.2f} TL\n"
            f"📊 <b>ROI:</b> %{roi*100:.2f}\n"
            f"✅ <b>Kazanılan:</b> {wins}\n"
            f"❌ <b>Kaybedilen:</b> {losses}\n"
            f"🏦 <b>Kasa:</b> {stats.get('bankroll', 0):.2f} TL\n\n"
            f"<i>Yarına hazırız.</i>"
        )
        return await self.send_message(msg)

    async def send_executive_summary(self, stats: Dict[str, Any]) -> bool:
        """CEO için detaylı günlük özet (Executive Summary)."""
        if not self.enabled:
            return False

        # Veri Hazırlığı
        bankroll = stats.get("bankroll", 0.0)
        daily_pnl = stats.get("daily_pnl", 0.0)
        roi = stats.get("roi", 0.0)
        drawdown = stats.get("drawdown", 0.0)
        win_rate = stats.get("win_rate", 0.0)
        total_bets = stats.get("total_bets", 0)

        # Trend Emojisi
        trend = "🚀" if daily_pnl > 0 else "🔻"
        if daily_pnl == 0:
            trend = "➖"

        # Risk Durumu
        risk_status = "GÜVENLİ"
        if drawdown < -0.10:
            risk_status = "DİKKAT"
        if drawdown < -0.20:
            risk_status = "KRİTİK"
        if stats.get("circuit_breaker", False):
            risk_status = "STOP-LOSS (KİLİTLİ)"

        msg = (
            f"<b>{trend} GÜNLÜK YÖNETİCİ ÖZETİ</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━\n\n"
            f"<b>💰 FİNANSAL DURUM</b>\n"
            f"• <b>Kasa:</b> {bankroll:,.2f} TL\n"
            f"• <b>Günlük PnL:</b> {daily_pnl:+,.2f} TL\n"
            f"• <b>ROI:</b> %{roi*100:.2f}\n\n"

            f"<b>🛡️ RİSK METRİKLERİ</b>\n"
            f"• <b>Drawdown:</b> %{drawdown*100:.2f}\n"
            f"• <b>Risk Statüsü:</b> {risk_status}\n\n"

            f"<b>📊 PERFORMANS</b>\n"
            f"• <b>Win Rate:</b> %{win_rate*100:.1f}\n"
            f"• <b>İşlem Hacmi:</b> {total_bets} Bahis\n\n"

            f"<i>\"Risk yönetimi, kârın temelidir.\" - JP Morgan</i>"
        )
        return await self.send_message(msg)
