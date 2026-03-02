"""
podcast_producer.py – Günlük bülten sesli radyo programı üretici.
Edge-TTS ile Türkçe sesli analiz bülteni oluşturur.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from loguru import logger

AUDIO_DIR = Path(__file__).resolve().parents[2] / "output" / "podcasts"


class PodcastProducer:
    """Sesli analiz bülteni üretici."""

    VOICE = "tr-TR-AhmetNeural"  # Türkçe erkek ses

    def __init__(self, voice: str = VOICE):
        self._voice = voice
        AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        logger.debug("PodcastProducer başlatıldı.")

    async def produce(self, signals: list[dict], bankroll: float = 10000) -> str:
        """Sinyallerden sesli bülten üretir."""
        script = self._write_script(signals, bankroll)
        output_path = AUDIO_DIR / f"bulten_{datetime.now().strftime('%Y%m%d_%H%M')}.mp3"

        try:
            import edge_tts
            communicate = edge_tts.Communicate(script, self._voice)
            await communicate.save(str(output_path))
            logger.info(f"Sesli bülten üretildi: {output_path}")
            return str(output_path)
        except ImportError:
            logger.warning("edge-tts yüklü değil – metin olarak kaydediliyor.")
            txt_path = output_path.with_suffix(".txt")
            txt_path.write_text(script, encoding="utf-8")
            return str(txt_path)
        except Exception as e:
            logger.error(f"TTS hatası: {e}")
            return ""

    def _write_script(self, signals: list[dict], bankroll: float) -> str:
        """Bülten metnini oluşturur."""
        now = datetime.now().strftime("%d %B %Y, %H:%M")
        active = [s for s in signals if s.get("selection") != "skip"]

        parts = [
            f"Quant Betting Bot günlük bülteni. {now}.",
            f"Şu an kasanız {bankroll:,.0f} TL.",
            f"Bugün toplamda {len(active)} adet aktif sinyal üretildi.",
        ]

        for i, sig in enumerate(active[:5], 1):
            match_id = sig.get("match_id", "Bilinmeyen maç")
            selection = sig.get("selection", "")
            ev = sig.get("ev", 0)
            conf = sig.get("confidence", 0)

            parts.append(
                f"Sinyal {i}: {match_id}. "
                f"Seçim: {selection}. "
                f"Beklenen değer: yüzde {ev*100:.1f}. "
                f"Güven seviyesi: yüzde {conf*100:.0f}."
            )

        if not active:
            parts.append("Bugün değer bahsi eşiğini geçen sinyal bulunamadı. Sabırlı olun.")

        parts.append("Bülten sonu. Bol şans.")
        return " ".join(parts)
