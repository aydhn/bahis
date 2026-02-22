"""
fbm_model.py – Fractional Brownian Motion (fBM) ve Hurst Eksponenti.

Standart modeller (Poisson, HMM) piyasanın "hafızasız" (Markov) olduğunu varsayar.
fBM ise piyasanın "uzun dönemli hafızasını" (Long Memory) ölçer.

Kavramlar:
  - Hurst Exponent (H):
     H < 0.5 -> Mean Reverting (Tersine dönen seri)
     H = 0.5 -> Random Walk (Beyaz gürültü)
     H > 0.5 -> Trending (Persistant/Kalıcı seri) - ASIL ALFA BURADA!
"""
from __future__ import annotations
import numpy as np
from typing import List, Dict, Any
from loguru import logger
from dataclasses import dataclass

@dataclass
class fBMResult:
    team: str
    hurst: float
    persistence: str # "persistant" | "anti-persistant" | "random"
    confidence: float

class fBMModel:
    def __init__(self, db: Any = None):
        self.db = db
        logger.debug("fBMModel başlatıldı.")

    def calculate_hurst(self, series: np.ndarray) -> float:
        """Rescaled Range (R/S) analizi ile Hurst eksponentini hesaplar."""
        if len(series) < 16: return 0.5
        
        n_vals = []
        rs_vals = []
        
        # Farklı pencere boyutları için R/S hesapla
        for n in [4, 8, 16, 32]:
            if n > len(series): break
            
            # Seriyi pencerelere böl
            segments = len(series) // n
            local_rs = []
            for i in range(segments):
                seg = series[i*n : (i+1)*n]
                # Mean-centered cumulative sum
                m = np.mean(seg)
                z = np.cumsum(seg - m)
                r = np.max(z) - np.min(z)
                s = np.std(seg)
                if s > 0:
                    local_rs.append(r / s)
            
            if local_rs:
                n_vals.append(np.log(n))
                rs_vals.append(np.log(np.mean(local_rs)))
                
        if len(n_vals) < 2: return 0.5
        
        # Log-Log dağılımının eğimi (Hurst)
        slope, _ = np.polyfit(n_vals, rs_vals, 1)
        return float(slope)

    def analyze_team(self, team: str, series: List[float]) -> fBMResult:
        arr = np.array(series)
        h = self.calculate_hurst(arr)
        
        persistence = "random"
        if h > 0.60: persistence = "persistant"
        elif h < 0.40: persistence = "anti-persistant"
        
        return fBMResult(
            team=team,
            hurst=round(h, 3),
            persistence=persistence,
            confidence=min(len(series) / 50, 1.0)
        )

    async def run_batch(self, **kwargs):
        """Pipeline entegrasyonu: Tüm takımların hafızasını ölçer."""
        logger.info("[fBM] Takım trend kalıcılığı analiz ediliyor...")
        # Mock: Normalde DB'den son performans serileri çekilir.
        return []
