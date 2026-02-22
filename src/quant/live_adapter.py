"""
live_adapter.py – Canlı Maç Adaptasyon ve Risk Motoru.

Maç içi dinamikleri (dakika, skor, kırmızı kart) analiz ederek 
model tahminlerini ve bahis miktarlarını gerçek zamanlı günceller.
"""
import numpy as np
from loguru import logger
from typing import Dict, Any

class LiveAdapter:
    def __init__(self):
        pass

    def calculate_time_decay(self, minute: str, base_lambda: float) -> float:
        """Kalan süreye göre gol beklentisini (lambda) hesaplar."""
        try:
            # "75'" formatını temizle
            min_int = int(minute.replace("'", "").split("+")[0])
            remaining_ratio = max(0, (90 - min_int) / 90)
            # Zaman ilerledikçe lambda (gol beklentisi) doğrusal azalır (Poisson Approximation)
            adjusted_lambda = base_lambda * remaining_ratio
            return adjusted_lambda
        except Exception:
            return base_lambda

    def adjust_kelly_live(self, current_stake: float, score_diff: int, minute: int) -> float:
        """Canlı skora göre stake (bahis miktarı) düzeltmesi."""
        # Eğer öndeysek (score_diff > 0) risk azalt
        if score_diff > 0 and minute > 70:
            return current_stake * 0.5 # Kar koruma modu
        # Eğer gerideysek ve dakika azsa risk minimize et
        if score_diff < 0 and minute > 80:
            return 0.0 # Stop-loss
        return current_stake

    def process_live_signal(self, pre_match_signal: dict, live_update: dict) -> dict:
        """Canlı veriyi sinyalle birleştirir."""
        minute = live_update.get("minute", "0'")
        score = live_update.get("score", "0-0")
        
        # Olasılıkları güncelle (Simule)
        pre_match_signal["live_score"] = score
        pre_match_signal["live_minute"] = minute
        pre_match_signal["confidence"] *= 0.95 # Canlıda belirsizlik çarpanı
        
        logger.debug(f"[LiveAdapter] Sinyal güncellendi: {score} ({minute})")
        return pre_match_signal
