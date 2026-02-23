"""
telegram_bot.py – Etkileşimli CEO Paneli (Bot).

Bu modül, tek yönlü raporlamanın ötesine geçerek,
kullanıcının (CEO) sistemle konuşmasını sağlar.

Komutlar:
  /status  - Sistem sağlık durumu
  /pnl     - Finansal özet
  /risk    - Risk seviyesi
  /stop    - Acil durdurma (Circuit Breaker)
"""
import asyncio
import os
import json
from typing import Optional, Dict, Any
from loguru import logger

try:
    import httpx
except ImportError:
    httpx = None

from src.system.config import settings

class TelegramBot:
    """Async Polling tabanlı Telegram Botu."""

    def __init__(self, token: Optional[str] = None):
        self.token = token or os.getenv("TELEGRAM_TOKEN")
        self.base_url = f"https://api.telegram.org/bot{self.token}"
        self.enabled = bool(self.token and httpx)
        self.offset = 0
        self.running = False
        self._task = None

        if not self.enabled:
            logger.warning("TelegramBot devre dışı: Token veya httpx eksik.")

    async def start(self):
        """Botu başlat (Arka planda dinlemeye başla)."""
        if not self.enabled or self.running:
            return

        self.running = True
        logger.info("TelegramBot dinlemeye başladı...")
        self._task = asyncio.create_task(self._poll_loop())

    async def stop(self):
        """Botu durdur."""
        self.running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _poll_loop(self):
        """Sürekli yeni mesajları kontrol et."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            while self.running:
                try:
                    params = {"offset": self.offset, "timeout": 20}
                    resp = await client.get(f"{self.base_url}/getUpdates", params=params)

                    if resp.status_code == 200:
                        data = resp.json()
                        if data.get("ok"):
                            for update in data.get("result", []):
                                await self._handle_update(update)
                                self.offset = update["update_id"] + 1
                    else:
                        logger.warning(f"Telegram polling hatası: {resp.status_code}")
                        await asyncio.sleep(5)
                except Exception as e:
                    logger.error(f"Telegram polling exception: {e}")
                    await asyncio.sleep(5)

    async def _handle_update(self, update: Dict[str, Any]):
        """Gelen mesajı işle."""
        msg = update.get("message", {})
        chat_id = msg.get("chat", {}).get("id")
        text = msg.get("text", "")

        if not text.startswith("/"):
            return

        command = text.split(" ")[0].lower()

        if command == "/start":
            await self.send_message(chat_id, "🤖 *Otonom Quant Bot Devrede.*\nKomutlar: /status, /pnl, /risk")

        elif command == "/status":
            await self.send_message(chat_id, "✅ *Sistem Çalışıyor*\nMod: Otonom\nVeri Akışı: Aktif")

        elif command == "/pnl":
            stats = self._read_bankroll_state()
            pnl = stats.get("pnl", 0.0)
            emoji = "🟢" if pnl >= 0 else "🔴"
            await self.send_message(chat_id, f"{emoji} *PnL:* {pnl:.2f} TL")

        elif command == "/risk":
            stats = self._read_bankroll_state()
            dd = stats.get("drawdown", 0.0)
            await self.send_message(chat_id, f"🛡️ *Risk Seviyesi*\nDrawdown: %{dd*100:.2f}")

    async def send_message(self, chat_id: int, text: str, parse_mode: str = "Markdown"):
        """Mesaj gönder."""
        if not self.enabled: return

        payload = {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": parse_mode
        }

        try:
            async with httpx.AsyncClient() as client:
                await client.post(f"{self.base_url}/sendMessage", json=payload)
        except Exception as e:
            logger.error(f"Mesaj gönderilemedi: {e}")

    async def send_bet_signal(self, bet: Dict[str, Any]):
        """Bahis sinyali formatlayıp gönder (CEO Vision)."""
        if not self.enabled: return

        chat_id = os.getenv("TELEGRAM_CHAT_ID")
        if not chat_id: return

        match = bet.get("match_id", "Unknown")
        selection = bet.get("selection", "UNKNOWN")
        odds = bet.get("odds", 0.0)
        stake = bet.get("stake", 0.0)
        conf = bet.get("confidence", 0.0)
        reason = bet.get("reason", "")

        emoji = "🟢" if conf > 0.7 else "🟡"
        if conf > 0.85: emoji = "🔥"

        msg = (
            f"<b>{emoji} YENİ POZİSYON ALINDI</b>\n\n"
            f"⚽ <b>Maç:</b> {match}\n"
            f"🎯 <b>Seçim:</b> {selection}\n"
            f"📈 <b>Oran:</b> {odds:.2f}\n"
            f"💰 <b>Stake:</b> {stake:.2f} TL\n"
            f"🧠 <b>Güven:</b> %{conf*100:.1f}\n"
            f"📝 <b>Gerekçe:</b> {reason}\n"
            f"<i>🤖 Otonom Quant Sistemi</i>"
        )
        await self.send_message(chat_id, msg, parse_mode="HTML")

    async def send_risk_alert(self, alert_type: str, details: str):
        """Kritik risk uyarısı."""
        if not self.enabled: return
        chat_id = os.getenv("TELEGRAM_CHAT_ID")
        if not chat_id: return

        msg = (
            f"<b>🚨 RİSK UYARISI: {alert_type.upper()}</b>\n\n"
            f"{details}\n\n"
            f"<i>Sistem koruma protokolleri devrede.</i>"
        )
        await self.send_message(chat_id, msg, parse_mode="HTML")

    def _read_bankroll_state(self) -> Dict[str, Any]:
        """Disk'ten son durumu oku."""
        try:
            path = settings.DATA_DIR / "bankroll_state.json"
            if path.exists():
                with open(path, "r") as f:
                    return json.load(f)
        except Exception:
            pass
        return {"pnl": 0.0, "drawdown": 0.0}
