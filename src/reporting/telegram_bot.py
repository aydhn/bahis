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

try:
    from src.ingestion.voice_interrogator import VoiceInterrogator
except ImportError:
    VoiceInterrogator = None

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
        self.voice_handler = VoiceInterrogator() if VoiceInterrogator else None
        self.bet_history = []  # Son bahisleri sakla (Explain için)

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

        # Sesli mesaj kontrolü
        if ("voice" in msg or "audio" in msg) and self.voice_handler:
            await self._handle_voice(msg, chat_id)
            return

        text = msg.get("text", "")

        if not text.startswith("/"):
            return

        parts = text.split(" ")
        command = parts[0].lower()
        args = parts[1:] if len(parts) > 1 else []

        if command == "/start":
            await self.send_message(chat_id, "🤖 *Otonom Quant Bot Devrede.*\nKomutlar: /status, /pnl, /risk, /explain")

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

        elif command == "/explain":
            await self._handle_explain(chat_id, args)

    async def _handle_voice(self, msg: Dict[str, Any], chat_id: int):
        """Sesli mesajı işle ve komuta çevir."""
        if not self.voice_handler: return

        file_id = msg.get("voice", {}).get("file_id") or msg.get("audio", {}).get("file_id")
        if not file_id: return

        await self.send_message(chat_id, "🎤 *Ses Analiz Ediliyor...*", parse_mode="Markdown")

        # Dosya yolunu al
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{self.base_url}/getFile?file_id={file_id}")
            file_path = resp.json().get("result", {}).get("file_path")

            if file_path:
                # İndir
                dl_url = f"https://api.telegram.org/file/bot{self.token}/{file_path}"
                audio_resp = await client.get(dl_url)

                # İşle
                result = await self.voice_handler.process_voice_message(audio_resp.content)
                cmd = result.get("command")

                if cmd != "unknown":
                    await self.send_message(chat_id, f"🗣️ Algılanan: *{result.get('raw')}* -> Komut: `/{cmd}`")
                    # Komutu simüle et
                    await self._handle_update({"message": {"chat": {"id": chat_id}, "text": f"/{cmd}"}})
                else:
                    await self.send_message(chat_id, f"🤔 Anlaşılamadı: *{result.get('raw')}*")

    async def _handle_explain(self, chat_id: int, args: list):
        """Son bahsin felsefi gerekçesini açıkla."""
        if not self.bet_history:
            await self.send_message(chat_id, "📭 Henüz açıklanacak bir bahis yok.")
            return

        # Varsa argüman olarak match_id, yoksa son bahis
        target_bet = self.bet_history[-1]

        philo = target_bet.get("philosophical_report")
        if not philo:
            await self.send_message(chat_id, "Bu bahis için felsefi rapor bulunamadı.")
            return

        # Raporu formatla
        report_msg = (
            f"🧠 <b>FELSEFİ ANALİZ RAPORU</b>\n\n"
            f"⚽ {target_bet.get('match_id')}\n"
            f"🎓 <b>Epistemik Skor:</b> {philo.epistemic_score:.2f} / 1.0\n\n"
            f"📉 <b>Black Swan Riski:</b> {philo.black_swan_risk:.2f}\n"
            f"💪 <b>Antifragility:</b> {philo.antifragility:.2f}\n"
            f"📚 <b>Lindy Skoru:</b> {philo.lindy_score:.2f}\n\n"
            f"💭 <b>Düşünceler:</b>\n"
        )
        for ref in philo.reflections:
            report_msg += f"- {ref}\n"

        await self.send_message(chat_id, report_msg, parse_mode="HTML")

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
        # Tarihçeye ekle
        self.bet_history.append(bet)
        if len(self.bet_history) > 10:
            self.bet_history.pop(0)

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
