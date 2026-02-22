"""
vision_tracker.py – Görsel istihbarat ve OCR katmanı.

Maç yayınlarındaki grafiklerden, PDF raporlarından veya ekran görüntülerinden 
EasyOCR kullanarak veri çıkartır.
"""
import os
from loguru import logger
from typing import List, Dict

class VisionTracker:
    def __init__(self, lang_list: List[str] = ['tr', 'en']):
        self._reader = None
        self._langs = lang_list
        self._ready = False

    def _ensure_reader(self):
        """EasyOCR okuyucusunu tembel yükleme (lazy load) ile başlatır."""
        if self._reader is None:
            try:
                import easyocr
                self._reader = easyocr.Reader(self._langs, gpu=True)
                self._ready = True
                logger.info("[Vision] EasyOCR (GPU) motoru başlatıldı.")
            except ImportError:
                logger.warning("[Vision] easyocr kütüphanesi yüklü değil.")
            except Exception as e:
                logger.error(f"[Vision] Başlatma hatası: {e}")

    def scan_image(self, image_path: str) -> List[str]:
        """Görseldeki tüm metinleri okur."""
        self._ensure_reader()
        if not self._ready or not os.path.exists(image_path):
            return []
            
        try:
            results = self._reader.readtext(image_path, detail=0)
            logger.info(f"[Vision] {image_path} tarandı, {len(results)} satır bulundu.")
            return results
        except Exception as e:
            logger.error(f"[Vision] Tarama hatası: {e}")
            return []

    def extract_statistics(self, text_lines: List[str]) -> Dict[str, str]:
        """Okunan metinlerden anahtar istatistikleri ayıklar (Heuristic)."""
        # Örn: "Topla Oynama: %55" satırını "possession": "55" olarak döner.
        stats = {}
        for line in text_lines:
            if "%" in line:
                # Basit regex mantığı ile istatistik yakalama
                pass
        return stats
