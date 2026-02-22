"""
voice_interrogator.py – Sesli Komut ve Sorgulama Asistanı.

Bu modül, Telegram üzerinden gelen sesli mesajları (Voice Notes)
text-to-speech (Whisper/SpeechRecognition) ile işleyip komutlara çevirir.
"""
from loguru import logger

class VoiceInterrogator:
    def __init__(self, model_name: str = "base"):
        self.model_name = model_name
        logger.info(f"VoiceInterrogator initialized with model: {model_name}")

    async def transcribe_voice(self, voice_file_path: str) -> str:
        """Ses dosyasını metne çevirir."""
        # Placeholder for Whisper or similar
        logger.info(f"Transcribing voice file: {voice_file_path}")
        return "status report" # Mock transcription

    async def process_intent(self, text: str) -> str:
        """Metinden komut niyetini çıkarır."""
        text = text.lower()
        if "rapor" in text or "report" in text:
            return "cmd_report"
        if "durum" in text or "status" in text:
            return "cmd_status"
        if "durdur" in text or "stop" in text:
            return "cmd_emergency_stop"
        return "cmd_unknown"
