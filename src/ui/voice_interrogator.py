"""
voice_interrogator.py – Yerel Whisper ile Sesli Komut İşleme.

Telegram'dan gelen ses mesajlarını metne çevirir ve otonom komutlara dönüştürür.
"""
import os
from pathlib import Path
from loguru import logger

try:
    import whisper
    WHISPER_OK = True
except ImportError:
    whisper = None
    WHISPER_OK = False

class VoiceInterrogator:
    def __init__(self, model_size: str = "base"):
        self.model_size = model_size
        self._model = None
        if WHISPER_OK:
            logger.info(f"[Voice] Whisper {model_size} modeli yükleniyor (CPU/GPU)...")
            # Lazy load
        else:
            logger.warning("[Voice] whisper kütüphanesi eksik. Sesli komut devredışı.")

    async def transcribe(self, audio_path: str) -> str:
        """Ses dosyasını metne çevirir."""
        if not WHISPER_OK:
            return "Whisper library not installed."
        
        try:
            if self._model is None:
                self._model = whisper.load_model(self.model_size)
            
            result = self._model.transcribe(audio_path)
            text = result.get("text", "").strip()
            logger.info(f"[Voice] Algılanan: {text}")
            return text
        except Exception as e:
            logger.error(f"[Voice] Transkripsiyon hatası: {e}")
            return ""

    def process_command(self, text: str) -> str:
        """Metni bota anlaşılır komuta çevirir (Simüle)."""
        text = text.lower()
        if "rapor" in text or "son durum" in text:
            return "/report"
        if "grafik" in text or "pnl" in text:
            return "/chart"
        if "benzer" in text:
            return "/similar"
        return text
