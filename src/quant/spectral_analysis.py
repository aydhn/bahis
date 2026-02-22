"""
spectral_analysis.py – Fast Fourier Transform (FFT) ile Döngü Analizi.

Futbol takımlarının performansları ve bahis oranları genellikle döngüseldir.
FFT kullanarak bu döngüleri (form grafiği periyotları) tespit ediyoruz.

Kavramlar:
  - FFT (Fast Fourier Transform): Zaman serisini frekans bileşenlerine ayırır.
  - Power Spectrum: Hangi frekansın (döngünün) ne kadar baskın olduğunu gösterir.
  - Dominant Period: En güçlü döngünün uzunluğu (örn. her 5 maçta bir düşüş).
  - Wavelet Denoising: Sinyal gürültüsünü temizlemek için (opsiyonel).

Kullanım:
    analyzer = SpectralAnalysis(db=db_manager)
    results = analyzer.analyze_team_cycles("Galatasaray", metric="performance_score")
    # -> {"dominant_period": 5.2, "cycle_strength": 0.85, ...}
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Any, List, Dict, Optional
from loguru import logger

@dataclass
class CycleResult:
    """Döngü analizi sonucu."""
    team: str
    metric: str
    dominant_period: float = 0.0  # Örn: 4.5 maç
    cycle_strength: float = 0.0   # 0.0 - 1.0 arası
    is_cyclic: bool = False
    next_peak_in: float = 0.0     # Tahmini zirveye kalan maç sayısı
    trend: str = "neutral"        # "up", "down", "neutral"

class SpectralAnalysis:
    """FFT tabanlı spektral analiz motoru."""

    def __init__(self, db: Any = None):
        self.db = db
        logger.debug("SpectralAnalysis başlatıldı.")

    def run_batch(self, **kwargs) -> List[Dict]:
        """Pipeline entegrasyonu için batch metodu."""
        logger.info("[SpectralAnalysis] Döngü analizi başlatılıyor...")
        
        if self.db is None:
            logger.warning("SpectralAnalysis: DB bağlantısı yok.")
            return []

        try:
            # 1. Aktif takımları bul
            teams_query = """
            SELECT DISTINCT home_team as name FROM matches 
            WHERE match_date >= CURRENT_DATE - INTERVAL '30 DAY'
            """
            teams_df = self.db.query(teams_query)
            if teams_df.is_empty():
                return []
            
            teams = teams_df["name"].tolist()
            results = []

            for team in teams:
                # 2. Takım performans serisini çek (Son 30 maç)
                # Basitlik için: (Gol Atılan - Gol Yenen) + 3 (galibiyet bonusu)
                perf_query = f"""
                SELECT 
                    CASE 
                        WHEN home_team = '{team}' THEN home_score - away_score + (CASE WHEN home_score > away_score THEN 3 ELSE 0 END)
                        ELSE away_score - home_score + (CASE WHEN away_score > home_score THEN 3 ELSE 0 END)
                    END as score
                FROM matches
                WHERE (home_team = '{team}' OR away_team = '{team}')
                AND status = 'finished'
                ORDER BY match_date DESC
                LIMIT 30
                """
                perf_df = self.db.query(perf_query)
                if len(perf_df) < 10:
                    continue
                
                # Tersten sırala (zaman akışına göre: eskiden yeniye)
                series = perf_df["score"].tolist()[::-1]
                res = self.analyze_series(series, team=team)
                results.append(res)
            
            return [vars(r) for r in results]

        except Exception as e:
            logger.error(f"SpectralAnalysis batch hatası: {e}")
            return []

    def analyze_series(self, series: List[float], team: str = "Unknown", metric: str = "score") -> CycleResult:
        """Bir zaman serisindeki baskın döngüyü bulur."""
        n = len(series)
        if n < 10:
            return CycleResult(team, metric)

        # 1. Trendi çıkar (Detrending)
        y = np.array(series)
        x = np.arange(n)
        
        try:
            p = np.polyfit(x, y, 1)
            trend_line = np.polyval(p, x)
            detrended = y - trend_line
        except Exception:
            detrended = y - np.mean(y)

        # 2. FFT Uygula
        # Hanning penceresi ile kenar etkilerini azalt
        windowed = detrended * np.hanning(n)
        
        fft_vals = np.fft.rfft(windowed)
        fft_freq = np.fft.rfftfreq(n)
        
        # Güç spektrumu (Power Spectrum)
        power = np.abs(fft_vals)**2
        
        # DC bileşenini (0 Hz) yoksay
        power[0] = 0
        
        # En baskın frekansı bul
        peak_idx = np.argmax(power)
        peak_freq = fft_freq[peak_idx]
        
        if peak_freq == 0:
            return CycleResult(team, metric)

        dominant_period = 1.0 / peak_freq
        
        # Döngü gücünü normalize et (Toplam güce oranı)
        total_power = np.sum(power)
        strength = power[peak_idx] / total_power if total_power > 0 else 0
        
        # Trend analizi (Son 3 maç)
        last_diff = series[-1] - series[-2]
        trend = "up" if last_diff > 0 else "down" if last_diff < 0 else "neutral"
        
        # Gelecek zirve tahmini (Basit faz analizi)
        # Sinyal dairesel olduğundan, son noktanın faza göre nerede olduğunu bul
        phase = np.angle(fft_vals[peak_idx])
        
        # Fazdan zirveye kalan mesafeyi hesapla (Radyan bazlı)
        # Karışıklık olmaması için basit bir 'due for peak' skoru
        # 0 radyan = Peak, pi radyan = Trough (yaklaşık)
        # Son noktanın fazını tahmin et: phase + 2*pi * (n-1)/period
        current_phase = (phase + 2 * np.pi * (n - 1) * peak_freq) % (2 * np.pi)
        
        # Zirveye uzaklık (maç sayısı olarak)
        # Faz 0'a ne kadar yakınsa o kadar zirvedeyiz.
        # Eğer current_phase > 0, bir sonraki 0'a 2*pi - current_phase mesafe var.
        dist_rad = (2 * np.pi - current_phase) % (2 * np.pi)
        next_peak_in = dist_rad / (2 * np.pi * peak_freq)
        
        return CycleResult(
            team=team,
            metric=metric,
            dominant_period=round(float(dominant_period), 2),
            cycle_strength=round(float(strength), 2),
            is_cyclic=strength > 0.25, # Eşik düşürüldü
            next_peak_in=round(float(next_peak_in), 1),
            trend=trend
        )

    def _mock_series(self, length: int = 20) -> List[float]:
        """Test için sinüs dalgalı rastgele veri üretir."""
        x = np.linspace(0, 4*np.pi, length)
        noise = np.random.normal(0, 0.2, length)
        # Periyodik sinyal + gürültü
        y = np.sin(x) + noise + 2.0 
        return y.tolist()
