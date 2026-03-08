"""
wavelet_denoiser.py – Wavelet Transforms (Sinyal Temizleme).

Bahis oranları sürekli iner çıkar. Bu iniş çıkışların çoğu
"gürültüdür". Wavelet Analizi gürültüyü siler ve oranların
altındaki "gerçek eğilimi" (signal) ortaya çıkarır.

Kavramlar:
  - Discrete Wavelet Transform (DWT): Sinyali farklı frekanslara
    ayırma — düşük frekans (trend) + yüksek frekans (gürültü)
  - Approximation Coefficients (cA): Düşük frekans bileşeni (trend)
  - Detail Coefficients (cD): Yüksek frekans bileşeni (gürültü)
  - Soft Thresholding: Gürültülü katsayıları sıfıra çekme
    (Donoho & Johnstone yöntemi)
  - Universal Threshold: λ = σ × √(2 × log(N))
  - Multi-Resolution Analysis (MRA): Farklı seviyelerde ayrıştırma
  - Mother Wavelet: db4 (Daubechies 4), sym5, coif3

Akış:
  1. Canlı oran zaman serisini al
  2. DWT ile çok seviyeli ayrıştırma (3-5 seviye)
  3. Yüksek frekanslı katsayılara Soft Thresholding
  4. Inverse DWT ile temiz sinyal oluştur
  5. Modellere temiz sinyal gönder (fake moves filtrelenmiş)

Teknoloji: PyWavelets (pywt)
Fallback: Basit moving average + exponential smoothing
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from loguru import logger

try:
    import pywt
    PYWT_OK = True
except ImportError:
    PYWT_OK = False
    logger.debug("pywt yüklü değil – moving average fallback.")


# ═══════════════════════════════════════════════
#  VERİ YAPILARI
# ═══════════════════════════════════════════════
@dataclass
class DenoiseResult:
    """Temizleme sonucu."""
    original: np.ndarray = field(default_factory=lambda: np.array([]))
    denoised: np.ndarray = field(default_factory=lambda: np.array([]))
    noise: np.ndarray = field(default_factory=lambda: np.array([]))
    # Metrikler
    snr_before: float = 0.0      # Signal-to-Noise Ratio (öncesi)
    snr_after: float = 0.0       # SNR (sonrası)
    noise_pct: float = 0.0       # Gürültü yüzdesi
    trend_direction: str = ""    # "up" | "down" | "flat"
    trend_slope: float = 0.0     # Eğim (temizlenmiş)
    # Meta
    wavelet: str = ""
    level: int = 0
    threshold: float = 0.0
    method: str = ""


@dataclass
class WaveletReport:
    """Dalgacık analizi raporu."""
    match_id: str = ""
    market: str = "odds"
    # Temizleme
    result: DenoiseResult = field(default_factory=DenoiseResult)
    # Frekans analizi
    energy_by_level: dict[int, float] = field(default_factory=dict)
    dominant_frequency: str = ""  # "low" (trend) | "mid" | "high" (gürültü)
    # Anomali
    fake_move_detected: bool = False
    fake_move_times: list[int] = field(default_factory=list)
    # Karar
    is_clean: bool = True
    recommendation: str = ""


# ═══════════════════════════════════════════════
#  TEMEL WAVELET FONKSİYONLARI
# ═══════════════════════════════════════════════
def soft_threshold(coeffs: np.ndarray, threshold: float) -> np.ndarray:
    """Soft Thresholding (Donoho & Johnstone).

    sign(x) * max(|x| - λ, 0)
    """
    return np.sign(coeffs) * np.maximum(np.abs(coeffs) - threshold, 0)


def universal_threshold(data: np.ndarray) -> float:
    """Universal Threshold: λ = σ × √(2 × log(N)).

    σ = MAD / 0.6745 (Median Absolute Deviation)
    """
    n = len(data)
    if n <= 1:
        return 0.0
    sigma = float(np.median(np.abs(data)) / 0.6745)
    return sigma * np.sqrt(2 * np.log(n))


def moving_average_denoise(data: np.ndarray, window: int = 5) -> np.ndarray:
    """Basit hareketli ortalama ile temizleme (fallback)."""
    if len(data) < window:
        return data.copy()
    kernel = np.ones(window) / window
    padded = np.pad(data, (window // 2, window // 2), mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed[:len(data)]


def exponential_smoothing(data: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """Exponential smoothing."""
    result = np.zeros_like(data)
    result[0] = data[0]
    for i in range(1, len(data)):
        result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]
    return result


# ═══════════════════════════════════════════════
#  WAVELET DENOISER (Ana Sınıf)
# ═══════════════════════════════════════════════
class WaveletDenoiser:
    """Dalgacık tabanlı sinyal temizleyici.

    Kullanım:
        wd = WaveletDenoiser(wavelet="db4", level=4)

        # Oran serisini temizle
        result = wd.denoise(odds_history)

        # Rapor
        report = wd.analyze(odds_history, match_id="gs_fb")

        # Toplu temizleme
        clean_data = wd.batch_denoise({"gs_fb": odds1, "bjk_ts": odds2})
    """

    AVAILABLE_WAVELETS = ["db4", "db6", "sym5", "coif3", "haar"]

    def __init__(self, wavelet: str = "db4", level: int = 4,
                 threshold_mode: str = "soft"):
        self._wavelet = wavelet
        self._level = level
        self._mode = threshold_mode
        self._last_result: DenoiseResult | None = None

        logger.debug(
            f"[Wavelet] Denoiser başlatıldı: {wavelet}, "
            f"level={level}, pywt={'OK' if PYWT_OK else 'fallback'}"
        )

    def denoise(self, data: np.ndarray | list) -> DenoiseResult:
        """Zaman serisini temizle."""
        result = DenoiseResult()
        signal = np.array(data, dtype=np.float64)
        signal = signal[np.isfinite(signal)]
        result.original = signal.copy()

        if len(signal) < 8:
            result.denoised = signal.copy()
            result.noise = np.zeros_like(signal)
            result.method = "too_short"
            return result

        if PYWT_OK:
            result = self._denoise_wavelet(signal, result)
        else:
            result = self._denoise_fallback(signal, result)

        self._last_result = result
        return result

    def _denoise_wavelet(self, signal: np.ndarray,
                           result: DenoiseResult) -> DenoiseResult:
        """PyWavelets ile temizleme."""
        try:
            max_level = pywt.dwt_max_level(len(signal), self._wavelet)
            level = min(self._level, max_level)

            # Çok seviyeli ayrıştırma
            coeffs = pywt.wavedec(signal, self._wavelet, level=level)

            # Threshold hesapla (detail katsayılarından)
            detail_concat = np.concatenate(coeffs[1:])
            threshold = universal_threshold(detail_concat)

            # Detail katsayılarını temizle (approximation dokunma)
            denoised_coeffs = [coeffs[0]]  # Approximation olduğu gibi
            for i, c in enumerate(coeffs[1:], 1):
                denoised_coeffs.append(soft_threshold(c, threshold))

            # Geri dönüştür
            denoised = pywt.waverec(denoised_coeffs, self._wavelet)
            denoised = denoised[:len(signal)]

            result.denoised = denoised
            result.noise = signal - denoised
            result.wavelet = self._wavelet
            result.level = level
            result.threshold = round(float(threshold), 6)
            result.method = "wavelet_dwt"

            # Enerji analizi (seviye bazlı)
            total_energy = sum(float(np.sum(c ** 2)) for c in coeffs)
            if total_energy > 0:
                for i, c in enumerate(coeffs):
                    energy = float(np.sum(c ** 2))
                    result_key = i  # 0 = approx, 1+ = details
                    # Seviye bazlı enerji oranını sakla (report'ta kullanılacak)

            # SNR
            signal_power = float(np.mean(signal ** 2))
            noise_power = float(np.mean(result.noise ** 2))
            if noise_power > 1e-10:
                result.snr_before = round(
                    10 * np.log10(signal_power / noise_power), 2,
                )
            result.noise_pct = round(
                float(noise_power / max(signal_power, 1e-10) * 100), 2,
            )

            # Temizlenmiş SNR
            denoised_power = float(np.mean(denoised ** 2))
            residual_noise = float(np.mean((signal - denoised) ** 2))
            if residual_noise > 1e-10:
                result.snr_after = round(
                    10 * np.log10(denoised_power / residual_noise), 2,
                )

            # Trend
            if len(denoised) >= 3:
                slope = (denoised[-1] - denoised[0]) / max(len(denoised), 1)
                result.trend_slope = round(float(slope), 6)
                if slope > 0.001:
                    result.trend_direction = "up"
                elif slope < -0.001:
                    result.trend_direction = "down"
                else:
                    result.trend_direction = "flat"

        except Exception as e:
            logger.debug(f"[Wavelet] DWT hatası: {e}")
            result = self._denoise_fallback(signal, result)

        return result

    def _denoise_fallback(self, signal: np.ndarray,
                            result: DenoiseResult) -> DenoiseResult:
        """Hareketli ortalama + exponential smoothing fallback."""
        # İki yöntemden en iyisini seç
        ma = moving_average_denoise(signal, window=7)
        es = exponential_smoothing(signal, alpha=0.2)

        # Hangisi daha düzgün?
        ma_var = float(np.var(np.diff(ma)))
        es_var = float(np.var(np.diff(es)))

        denoised = ma if ma_var < es_var else es
        result.denoised = denoised
        result.noise = signal - denoised
        result.method = "moving_average" if ma_var < es_var else "exponential_smoothing"

        # SNR
        signal_power = float(np.mean(signal ** 2))
        noise_power = float(np.mean(result.noise ** 2))
        if noise_power > 1e-10:
            result.snr_before = round(
                10 * np.log10(signal_power / noise_power), 2,
            )
        result.noise_pct = round(
            float(noise_power / max(signal_power, 1e-10) * 100), 2,
        )

        # Trend
        if len(denoised) >= 3:
            slope = (denoised[-1] - denoised[0]) / max(len(denoised), 1)
            result.trend_slope = round(float(slope), 6)
            if slope > 0.001:
                result.trend_direction = "up"
            elif slope < -0.001:
                result.trend_direction = "down"
            else:
                result.trend_direction = "flat"

        return result

    def analyze(self, data: np.ndarray | list,
                  match_id: str = "",
                  market: str = "odds") -> WaveletReport:
        """Tam dalgacık analiz raporu."""
        report = WaveletReport(match_id=match_id, market=market)

        result = self.denoise(data)
        report.result = result

        # Enerji analizi
        signal = np.array(data, dtype=np.float64)
        signal = signal[np.isfinite(signal)]

        if PYWT_OK and len(signal) >= 8:
            try:
                max_level = pywt.dwt_max_level(len(signal), self._wavelet)
                level = min(self._level, max_level)
                coeffs = pywt.wavedec(signal, self._wavelet, level=level)

                total_energy = sum(float(np.sum(c ** 2)) for c in coeffs)
                for i, c in enumerate(coeffs):
                    energy = float(np.sum(c ** 2))
                    pct = round(energy / max(total_energy, 1e-10) * 100, 1)
                    report.energy_by_level[i] = pct

                # Baskın frekans
                approx_energy = report.energy_by_level.get(0, 0)
                if approx_energy > 70:
                    report.dominant_frequency = "low"
                elif approx_energy > 40:
                    report.dominant_frequency = "mid"
                else:
                    report.dominant_frequency = "high"
            except Exception:
                pass

        # Fake move tespiti
        if len(result.noise) > 0:
            noise_std = float(np.std(result.noise))
            for i, n in enumerate(result.noise):
                if abs(n) > 3 * noise_std:
                    report.fake_move_detected = True
                    report.fake_move_times.append(i)

        report.is_clean = not report.fake_move_detected
        report.recommendation = self._advice(report)
        return report

    def batch_denoise(self, series_dict: dict[str, list | np.ndarray]) -> dict[str, np.ndarray]:
        """Toplu temizleme."""
        results = {}
        for key, data in series_dict.items():
            result = self.denoise(data)
            results[key] = result.denoised
        return results

    def _advice(self, r: WaveletReport) -> str:
        res = r.result
        if r.fake_move_detected:
            n_fakes = len(r.fake_move_times)
            return (
                f"FAKE MOVE TESPİT: {n_fakes} sahte hareket! "
                f"t={r.fake_move_times[:3]}. "
                f"Gürültü: %{res.noise_pct:.1f}, "
                f"SNR: {res.snr_before:.1f}→{res.snr_after:.1f}dB. "
                f"Temizlenmiş trend: {res.trend_direction} "
                f"({res.trend_slope:+.4f})."
            )
        return (
            f"TEMİZ SİNYAL: Gürültü %{res.noise_pct:.1f}, "
            f"trend: {res.trend_direction} ({res.trend_slope:+.4f}). "
            f"Wavelet: {res.wavelet or res.method}, "
            f"SNR: {res.snr_before:.1f}→{res.snr_after:.1f}dB."
        )
