"""
voice_notifier.py – Sesli Bildirim ve TTS (Text-to-Speech) Modülü.

Kullanıcı ekran başında değilken kritik sinyalleri kaçırmaması için 
Telegram üzerinden sesli mesaj (voice note) gönderir.
"""
import os
import asyncio
from typing import Dict, Any
from loguru import logger
try:
    from gtts import gTTS
    TTS_OK = True
except ImportError:
    TTS_OK = False
    logger.warning("gTTS kütüphanesi yüklü değil. Sesli bildirimler devre dışı.")

class VoiceNotifier:
    def __init__(self, temp_dir: str = "data/voice"):
        self.temp_dir = temp_dir
        os.makedirs(temp_dir, exist_ok=True)

    async def generate_voice(self, text: str, filename: str = "alert.mp3") -> str:
        """Metni sese çevirir ve dosya yolunu döner."""
        if not TTS_OK:
            return ""

        path = os.path.join(self.temp_dir, filename)
        try:
            # gTTS senkron olduğu için thread'de çalıştır
            tts = gTTS(text=text, lang='tr')
            await asyncio.to_thread(tts.save, path)
            return path
        except Exception as e:
            logger.error(f"TTS hatası: {e}")
            return ""

    async def send_voice_alert(self, bot_app: Any, chat_id: str, signal: Dict[str, Any]):
        """Sinyali sesli mesaj olarak gönderir."""
        if not TTS_OK:
            return

        text = (
            f"Yeni sinyal! {signal.get('match_id', 'Maç')} için "
            f"{signal.get('selection', 'bahis')} seçeneği "
            f"{signal.get('odds', 1.0)} oran ile mevcut. "
            f"Güven skoru yüzde {int(signal.get('confidence', 0) * 100)}."
        )
        
        voice_path = await self.generate_voice(text)
        if voice_path and os.path.exists(voice_path):
            try:
                with open(voice_path, 'rb') as voice:
                    await bot_app.bot.send_voice(chat_id=chat_id, voice=voice)
                logger.info(f"Sesli bildirim gönderildi: {signal.get('match_id')}")
            except Exception as e:
                logger.error(f"Telegram voice gönderme hatası: {e}")
            finally:
                # Geçici dosyayı temizle
                if os.path.exists(voice_path):
                    os.remove(voice_path)
