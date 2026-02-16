"""
voice_interrogator.py – Whisper ile sesli Telegram komutlarını metne çevirir.
Sesli mesajları alır, transkript eder, komut olarak yorumlar.
"""
from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from loguru import logger


class VoiceInterrogator:
    """Sesli mesajları metne çevirip komut olarak yorumlar."""

    COMMANDS = {
        "rapor": "report",
        "durum": "status",
        "bahis": "bets",
        "kapat": "stop",
        "başlat": "start",
        "analiz": "analyze",
        "eşik": "threshold",
        "risk": "risk",
    }

    def __init__(self, model_size: str = "base"):
        self._model = None
        self._model_size = model_size
        logger.debug("VoiceInterrogator başlatıldı.")

    def _load_model(self):
        if self._model is not None:
            return
        try:
            import whisper
            self._model = whisper.load_model(self._model_size)
            logger.info(f"Whisper modeli yüklendi: {self._model_size}")
        except ImportError:
            logger.warning("openai-whisper yüklü değil – ses tanıma devre dışı.")
        except Exception as e:
            logger.error(f"Whisper yükleme hatası: {e}")

    def transcribe(self, audio_path: str | Path) -> str:
        """Ses dosyasını metne çevirir."""
        self._load_model()
        if self._model is None:
            return ""
        try:
            result = self._model.transcribe(str(audio_path), language="tr")
            text = result.get("text", "").strip()
            logger.info(f"Transkript: {text[:100]}…")
            return text
        except Exception as e:
            logger.error(f"Transkripsiyon hatası: {e}")
            return ""

    def parse_command(self, text: str) -> dict:
        """Metinden komut çıkarır."""
        text_lower = text.lower().strip()
        for keyword, cmd in self.COMMANDS.items():
            if keyword in text_lower:
                # Parametreleri çıkar
                parts = text_lower.split(keyword, 1)
                params = parts[1].strip() if len(parts) > 1 else ""
                return {"command": cmd, "params": params, "raw": text}
        return {"command": "unknown", "params": "", "raw": text}

    async def process_voice_message(self, audio_bytes: bytes) -> dict:
        """Telegram'dan gelen ses baytlarını işler."""
        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        text = self.transcribe(tmp_path)
        command = self.parse_command(text)

        # Geçici dosyayı sil
        try:
            Path(tmp_path).unlink()
        except OSError:
            pass

        return command

    async def listen(self, shutdown: asyncio.Event):
        """Sesli komut dinleyici döngüsü (Telegram entegrasyonu ile)."""
        logger.info("VoiceInterrogator – dinleyici hazır (Telegram entegrasyonu bekliyor).")
        while not shutdown.is_set():
            await asyncio.sleep(10)
