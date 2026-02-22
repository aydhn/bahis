"""
voice_engine.py – Voice of God (Ses Motoru).

Amacı:
Metin tabanlı uyarıları konuşmaya çevirmek.
Telegram'a ses kaydı (.mp3/.ogg) göndermek için kullanılır.

Teknoloji:
- pyttsx3: Çevrimdışı TTS motoru. (Windows SAPI5 / Linux eSpeak)
"""
import os
import time
from pathlib import Path
from loguru import logger

try:
    import pyttsx3
    HAS_TTS = True
except ImportError:
    HAS_TTS = False
    logger.warning("[VoiceEngine] pyttsx3 yüklü değil, ses üretilemeyecek.")

class VoiceEngine:
    def __init__(self, output_dir: str = "data/voice_notes"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.engine = None
        if HAS_TTS:
            try:
                self.engine = pyttsx3.init()
                # Ayarlar
                self.engine.setProperty('rate', 145)   # Konuşma hızı
                self.engine.setProperty('volume', 1.0) # Ses
                
                # Türkçe ses bulmaya çalış
                voices = self.engine.getProperty('voices')
                for voice in voices:
                    if "turkish" in voice.name.lower() or "tr" in voice.id.lower():
                        self.engine.setProperty('voice', voice.id)
                        break
            except Exception as e:
                logger.error(f"[VoiceEngine] Başlatma hatası: {e}")
                self.engine = None

    def generate_audio(self, text: str, filename: str = "alert.mp3") -> str:
        """Metni sese çevirir ve dosyaya kaydeder."""
        if not self.engine:
            return ""

        output_path = self.output_dir / filename
        
        # pyttsx3 save_to_file asenkron çalışmaz, bloklar.
        # Kısa metinler için sorun değil.
        try:
            self.engine.save_to_file(text, str(output_path))
            self.engine.runAndWait() # İşlemi bitir ve bekle
            
            if output_path.exists() and output_path.stat().st_size > 0:
                logger.info(f"[VoiceEngine] Ses oluşturuldu: {output_path}")
                return str(output_path)
            else:
                logger.error("[VoiceEngine] Dosya oluşturulamadı.")
                return ""
        except Exception as e:
            logger.error(f"[VoiceEngine] Üretim hatası: {e}")
            return ""
