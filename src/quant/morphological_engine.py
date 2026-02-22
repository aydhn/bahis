"""
morphological_engine.py – Morfolojik sinyal işleme ve örüntü tanıma.

Zaman serisi verilerini (oranlar, skorlar) morfolojik operatörler kullanarak 
analiz eder. Trendleri ve dönüşleri görsel bir örüntü olarak işler.
"""
import numpy as np
from scipy import ndimage
from loguru import logger
from typing import List

class MorphologicalEngine:
    def __init__(self, window_size: int = 15):
        self._window = window_size

    def clean_signal(self, data: np.ndarray) -> np.ndarray:
        """Sinyaldeki gürültüyü morfolojik 'opening' (aşındırma + genişletme) ile temizler."""
        # Veriyi 1D dizi olarak ele al
        # Erosion -> Dilation
        eroded = ndimage.grey_erosion(data, size=self._window)
        opened = ndimage.grey_dilation(eroded, size=self._window)
        return opened

    def detect_local_extrema(self, data: np.ndarray) -> List[int]:
        """Sinyaldeki yerel zirve ve dipleri (extrema) tespit eder."""
        # Top-hat transformasyonu ile trendden arındırma
        top_hat = data - ndimage.grey_dilation(ndimage.grey_erosion(data, size=self._window), size=self._window)
        
        # Sınırları aşan noktalar ekstrem noktalardır
        threshold = np.std(top_hat) * 2.0
        extrema = np.where(np.abs(top_hat) > threshold)[0]
        return extrema.tolist()

    def process_odds_movement(self, odds_history: List[float]) -> str:
        """Oran hareketini analiz eder ve sinyal üretir."""
        if len(odds_history) < self._window:
            return "NEUTRAL"
            
        data = np.array(odds_history)
        cleaned = self.clean_signal(data)
        
        # Son değişim trendi
        diff = cleaned[-1] - cleaned[-2]
        if diff > 0: return "RISING"
        if diff < 0: return "FALLING"
        return "STABLE"
