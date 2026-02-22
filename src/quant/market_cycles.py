"""
market_cycles.py – FFT (Fast Fourier Transform) Tabanlı Döngü Analizi.

Bu modül, zaman serisi verilerinde (oranlar, sonuçlar, xG farkları)
gizli periyodik kalıpları (döngüleri) tespit eder.

Kullanım Alanı:
- "Bu takım her 4 maçta bir patlama yapıyor."
- "Ligde her 7 haftada bir favorilerin kaybetme oranı artıyor."
"""
import numpy as np
# import polars as pl # Polars henüz kullanılmıyor, kaldırıldı
from scipy.fft import fft, fftfreq
from dataclasses import dataclass
from loguru import logger

@dataclass
class CycleResult:
    period: float       # Döngü periyodu (ör: 4.2 maç)
    strength: float     # Sinyal gücü (genlik)
    confidence: float   # İstatistiksel güven
    description: str    # İnsan tarafından okunabilir açıklama

class MarketCycles:
    """Piyasa döngülerini analiz eden FFT motoru."""
    
    def __init__(self, min_period: float = 3.0):
        self._min_period = min_period
        
    def analyze_series(self, data: list[float], sample_rate: float = 1.0) -> list[CycleResult]:
        """Bir zaman serisindeki baskın döngüleri bulur.
        
        Args:
           data: Analiz edilecek veri listesi (ör: son 50 maçın xG farkı)
           sample_rate: Örnekleme hızı (maç başına 1 ise 1.0)
        """
        n = len(data)
        if n < 10:
            return []
            
        # DC component (ortalama) çıkarılır, detrend yapılır
        signal = np.array(data)
        signal = signal - np.mean(signal)
        
        # FFT Uygula
        yf = fft(signal)
        xf = fftfreq(n, 1 / sample_rate)
        
        # Sadece pozitif frekanslar
        mask = xf > 0
        freqs = xf[mask]
        magnitudes = np.abs(yf[mask])
        
        # Normalize et
        magnitudes = magnitudes / n
        
        # Tepe noktalarını bul (Dominant cycles)
        # Basit eşik: Ortalama genliğin 2 katı
        threshold = np.mean(magnitudes) * 2.0
        peaks_indices = np.where(magnitudes > threshold)[0]
        
        results = []
        for idx in peaks_indices:
            freq = freqs[idx]
            mag = magnitudes[idx]
            period = 1 / freq
            
            if period < self._min_period:
                continue
                
            confidence = min(mag / np.max(magnitudes), 1.0)
            
            res = CycleResult(
                period=float(period),
                strength=float(mag),
                confidence=float(confidence),
                description=f"Her {period:.1f} birimde bir döngü (Güç: {mag:.2f})"
            )
            results.append(res)
            
        # Güç sırasına göre diz
        results.sort(key=lambda x: x.strength, reverse=True)
        return results

    def analyze_team_form(self, results: list[int]) -> str:
        """Takımın galibiyet/mağlubiyet serisindeki döngüleri analiz eder.
        1: Win, 0: Draw/Loss
        """
        cycles = self.analyze_series(results)
        if not cycles:
            return "Döngü tespit edilemedi (Rastgele dağılım)."
            
        top = cycles[0]
        return f"🎯 Tespit: Takım performansında {top.period:.1f} maçlık bir döngü var (Güven: %{top.confidence*100:.0f})."
