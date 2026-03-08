"""
multifractal_logic.py – MF-DFA (Multifractal Detrended Fluctuation Analysis).

Piyasa sadece fraktal değildir, Çoklu Fraktaldır (Multifractal).
Sakin zamanlardaki kaos ile kriz anlarındaki kaos farklı matematiklere
sahiptir.

Kavramlar:
  - DFA (Detrended Fluctuation Analysis): Zaman serisindeki uzun-menzilli
    korelasyonları ölçer
  - MF-DFA: DFA'nın genelleştirilmiş hali — farklı momentler (q) için
    farklı Hurst üsleri hesaplar
  - Hurst Exponent h(q): q-bağımlı ölçekleme üssü
  - Multifractal Spectrum: τ(q) → f(α) dönüşümü, α = dh/dq
  - Singularity Spectrum: f(α) vs α grafiği — "genişlik" = çoklu fraktallik
  - Mono-fractal: h(q) sabit → piyasa homojen
  - Multi-fractal: h(q) değişken → piyasa heterojen, rejim değişimi yakın
  - Δh = h(q_min) - h(q_max) → çoklu fraktallik derecesi

Akış:
  1. Oran zaman serisini al (son N dakika/saat)
  2. Kümülatif sapma serisi oluştur (random walk profile)
  3. Farklı pencere boyutları (s) için DFA uygula
  4. Her q değeri için log-log regresyon → h(q)
  5. Singularity spectrum f(α) hesapla
  6. Δh büyükse → rejim değişikliği uyarısı

Teknoloji: MFDFA veya özel numpy implementasyonu
Fallback: Manuel DFA + polinom fit
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from loguru import logger

try:
    from MFDFA import MFDFA as _MFDFA_func
    MFDFA_LIB_OK = True
except ImportError:
    MFDFA_LIB_OK = False
    logger.debug("MFDFA kütüphanesi yüklü değil – manuel DFA fallback.")

try:
    from scipy.stats import linregress
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False


# ═══════════════════════════════════════════════
#  VERİ YAPILARI
# ═══════════════════════════════════════════════
@dataclass
class MFDFAParams:
    """MF-DFA parametreleri."""
    hurst_q2: float = 0.5         # Standart Hurst (q=2)
    delta_h: float = 0.0          # h(q_min) - h(q_max) = çoklu fraktallik
    h_values: dict[float, float] = field(default_factory=dict)  # q → h(q)
    # Singularity spectrum
    alpha_min: float = 0.0
    alpha_max: float = 0.0
    alpha_width: float = 0.0      # α genişliği = çoklu fraktallik
    f_alpha_max: float = 0.0      # f(α) tepe değeri


@dataclass
class MultifractalReport:
    """Çoklu fraktal analiz raporu."""
    match_id: str = ""
    market: str = "odds"
    # Parametreler
    params: MFDFAParams = field(default_factory=MFDFAParams)
    # Rejim
    regime: str = ""           # "monofractal" | "weak_multifractal" | "strong_multifractal"
    regime_change_signal: bool = False
    # Trend
    is_trending: bool = False  # h(2) > 0.5
    is_mean_reverting: bool = False  # h(2) < 0.5
    persistence: float = 0.0  # |h(2) - 0.5|
    # Meta
    n_points: int = 0
    method: str = ""
    recommendation: str = ""


# ═══════════════════════════════════════════════
#  MANUEL DFA (Detrended Fluctuation Analysis)
# ═══════════════════════════════════════════════
def cumulative_profile(x: np.ndarray) -> np.ndarray:
    """Kümülatif sapma profili: Y(i) = Σ(x_k - <x>)."""
    return np.cumsum(x - np.mean(x))


def dfa_fluctuation(profile: np.ndarray, scale: int) -> float:
    """Tek bir pencere boyutu için DFA dalgalanma fonksiyonu.

    F(s) = sqrt(1/N * Σ (Y - trend)^2)
    """
    n = len(profile)
    n_segments = n // scale
    if n_segments == 0:
        return 0.0

    fluctuations = []
    for v in range(n_segments):
        segment = profile[v * scale:(v + 1) * scale]
        x_range = np.arange(len(segment))
        # Doğrusal trend çıkarma
        if SCIPY_OK:
            slope, intercept, _, _, _ = linregress(x_range, segment)
            trend = slope * x_range + intercept
        else:
            coeffs = np.polyfit(x_range, segment, 1)
            trend = np.polyval(coeffs, x_range)

        rms = np.sqrt(np.mean((segment - trend) ** 2))
        fluctuations.append(rms)

    return float(np.mean(fluctuations)) if fluctuations else 0.0


def generalized_dfa(profile: np.ndarray, scale: int, q: float) -> float:
    """Genelleştirilmiş DFA: q-moment dalgalanma fonksiyonu.

    F_q(s) = {1/N_s * Σ [F²(s,v)]^(q/2)}^(1/q)
    """
    n = len(profile)
    n_segments = n // scale
    if n_segments == 0:
        return 0.0

    f_squared = []
    for v in range(n_segments):
        segment = profile[v * scale:(v + 1) * scale]
        x_range = np.arange(len(segment))
        if SCIPY_OK:
            slope, intercept, _, _, _ = linregress(x_range, segment)
            trend = slope * x_range + intercept
        else:
            coeffs = np.polyfit(x_range, segment, 1)
            trend = np.polyval(coeffs, x_range)

        variance = np.mean((segment - trend) ** 2)
        f_squared.append(max(variance, 1e-20))

    f_sq = np.array(f_squared)

    if abs(q) < 1e-6:
        return float(np.exp(0.5 * np.mean(np.log(f_sq))))

    if q > 0:
        return float(np.mean(f_sq ** (q / 2)) ** (1 / q))
    else:
        vals = f_sq ** (q / 2)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            return 0.0
        return float(np.mean(vals) ** (1 / q))


def compute_mfdfa(data: np.ndarray,
                    q_range: np.ndarray | None = None,
                    min_scale: int = 8,
                    max_scale: int | None = None) -> MFDFAParams:
    """Manuel MF-DFA hesaplama."""
    params = MFDFAParams()
    n = len(data)

    if n < 50:
        return params

    if q_range is None:
        q_range = np.arange(-5, 5.1, 0.5)
        q_range = q_range[q_range != 0]  # q=0 ayrı işlenir

    if max_scale is None:
        max_scale = n // 4

    profile = cumulative_profile(data)

    # Ölçek aralığı
    scales = np.unique(np.logspace(
        np.log10(min_scale),
        np.log10(max_scale),
        num=20,
    ).astype(int))
    scales = scales[(scales >= min_scale) & (scales <= max_scale)]
    if len(scales) < 3:
        return params

    # Her q için h(q) hesapla
    h_values = {}
    for q in q_range:
        fluct = []
        valid_scales = []
        for s in scales:
            f = generalized_dfa(profile, int(s), q)
            if f > 0:
                fluct.append(np.log(f))
                valid_scales.append(np.log(s))

        if len(valid_scales) >= 3:
            if SCIPY_OK:
                slope, _, _, _, _ = linregress(valid_scales, fluct)
            else:
                coeffs = np.polyfit(valid_scales, fluct, 1)
                slope = coeffs[0]
            h_values[float(q)] = round(float(slope), 4)

    if not h_values:
        return params

    params.h_values = h_values
    params.hurst_q2 = h_values.get(2.0, 0.5)

    # Çoklu fraktallik derecesi
    h_vals = list(h_values.values())
    params.delta_h = round(float(max(h_vals) - min(h_vals)), 4)

    # Singularity spectrum (Legendre dönüşümü)
    q_sorted = sorted(h_values.keys())
    if len(q_sorted) >= 3:
        tau = np.array([q * h_values[q] - 1 for q in q_sorted])
        q_arr = np.array(q_sorted)

        # α = dτ/dq (sayısal türev)
        alpha = np.gradient(tau, q_arr)
        f_alpha = q_arr * alpha - tau

        valid = np.isfinite(alpha) & np.isfinite(f_alpha)
        if valid.any():
            alpha = alpha[valid]
            f_alpha = f_alpha[valid]

            params.alpha_min = round(float(np.min(alpha)), 4)
            params.alpha_max = round(float(np.max(alpha)), 4)
            params.alpha_width = round(
                float(params.alpha_max - params.alpha_min), 4,
            )
            params.f_alpha_max = round(float(np.max(f_alpha)), 4)

    return params


# ═══════════════════════════════════════════════
#  MULTIFRACTAL LOGIC (Ana Sınıf)
# ═══════════════════════════════════════════════
class MultifractalAnalyzer:
    """Çoklu fraktal piyasa analizi.

    Kullanım:
        mfa = MultifractalAnalyzer()

        # Oran zaman serisini analiz et
        report = mfa.analyze(odds_history, match_id="gs_fb", market="odds")

        # Rejim değişikliği erken uyarısı
        if report.regime_change_signal:
            logger.warning("Rejim değişikliği yaklaşıyor!")
    """

    DELTA_H_WEAK = 0.2       # Zayıf çoklu fraktallik
    DELTA_H_STRONG = 0.5     # Güçlü çoklu fraktallik
    ALPHA_WIDTH_CRITICAL = 0.8  # Kritik singularity genişliği

    def __init__(self, q_min: float = -5, q_max: float = 5,
                 q_step: float = 0.5, min_scale: int = 8):
        self._q_range = np.arange(q_min, q_max + q_step, q_step)
        self._q_range = self._q_range[self._q_range != 0]
        self._min_scale = min_scale
        self._history: list[MultifractalReport] = []

        logger.debug(
            f"[MF-DFA] Başlatıldı: q=[{q_min}, {q_max}], "
            f"step={q_step}, min_scale={min_scale}"
        )

    def analyze(self, time_series: list[float] | np.ndarray,
                  match_id: str = "",
                  market: str = "odds") -> MultifractalReport:
        """Çoklu fraktal analiz."""
        report = MultifractalReport(match_id=match_id, market=market)
        data = np.array(time_series, dtype=np.float64)

        # NaN/Inf temizle
        data = data[np.isfinite(data)]
        report.n_points = len(data)

        if len(data) < 50:
            report.regime = "insufficient_data"
            report.recommendation = (
                f"Yetersiz veri ({len(data)} nokta, min 50 gerekli)."
            )
            report.method = "none"
            return report

        # Returns'e çevir (log-returns)
        if np.all(data > 0):
            returns = np.diff(np.log(data))
        else:
            returns = np.diff(data)

        if len(returns) < 30:
            report.regime = "insufficient_data"
            report.recommendation = "Yetersiz return verisi."
            report.method = "none"
            return report

        # MF-DFA kütüphanesi
        if MFDFA_LIB_OK:
            try:
                report = self._analyze_mfdfa_lib(returns, report)
                report.method = "mfdfa_library"
            except Exception as e:
                logger.debug(f"[MF-DFA] Kütüphane hatası: {e}")
                report = self._analyze_manual(returns, report)
                report.method = "manual_mfdfa"
        else:
            report = self._analyze_manual(returns, report)
            report.method = "manual_mfdfa"

        # Rejim sınıflandırma
        report = self._classify_regime(report)
        report.recommendation = self._advice(report)
        self._history.append(report)

        return report

    def _analyze_mfdfa_lib(self, returns: np.ndarray,
                              report: MultifractalReport) -> MultifractalReport:
        """MFDFA kütüphanesiyle analiz."""
        scales = np.unique(np.logspace(
            np.log10(self._min_scale),
            np.log10(len(returns) // 4),
            num=20,
        ).astype(int))

        lag, fluct = _MFDFA_func(
            returns, lag=scales, q=self._q_range,
        )

        params = MFDFAParams()
        h_values = {}

        for i, q in enumerate(self._q_range):
            valid = (fluct[:, i] > 0) & (lag > 0)
            if valid.sum() >= 3:
                if SCIPY_OK:
                    slope, _, _, _, _ = linregress(
                        np.log(lag[valid]), np.log(fluct[valid, i]),
                    )
                else:
                    coeffs = np.polyfit(
                        np.log(lag[valid]), np.log(fluct[valid, i]), 1,
                    )
                    slope = coeffs[0]
                h_values[float(q)] = round(float(slope), 4)

        params.h_values = h_values
        params.hurst_q2 = h_values.get(2.0, 0.5)

        h_vals = list(h_values.values())
        if h_vals:
            params.delta_h = round(float(max(h_vals) - min(h_vals)), 4)

        report.params = params
        return report

    def _analyze_manual(self, returns: np.ndarray,
                          report: MultifractalReport) -> MultifractalReport:
        """Manuel MF-DFA."""
        params = compute_mfdfa(
            returns, q_range=self._q_range, min_scale=self._min_scale,
        )
        report.params = params
        return report

    def _classify_regime(self, report: MultifractalReport) -> MultifractalReport:
        """Rejim sınıflandırma."""
        p = report.params

        # Trend/mean-reversion
        report.persistence = round(abs(p.hurst_q2 - 0.5), 4)
        report.is_trending = p.hurst_q2 > 0.55
        report.is_mean_reverting = p.hurst_q2 < 0.45

        # Çoklu fraktallik
        if p.delta_h < self.DELTA_H_WEAK:
            report.regime = "monofractal"
        elif p.delta_h < self.DELTA_H_STRONG:
            report.regime = "weak_multifractal"
        else:
            report.regime = "strong_multifractal"

        # Rejim değişikliği sinyali
        if len(self._history) >= 2:
            prev = self._history[-1]
            if (prev.regime == "monofractal"
                    and report.regime in ("weak_multifractal", "strong_multifractal")):
                report.regime_change_signal = True
            elif (prev.params.delta_h > 0 and p.delta_h > 0):
                change_rate = (p.delta_h - prev.params.delta_h) / max(prev.params.delta_h, 0.01)
                if change_rate > 0.5:
                    report.regime_change_signal = True

        return report

    def _advice(self, r: MultifractalReport) -> str:
        p = r.params
        if r.regime_change_signal:
            return (
                f"REJİM DEĞİŞİKLİĞİ UYARISI: Piyasa mono→multi fraktale geçiyor! "
                f"Δh={p.delta_h:.3f}, h(2)={p.hurst_q2:.3f}. "
                f"Büyük sürpriz/çöküş riski. TÜM BAHİSLERİ DURDUR."
            )
        if r.regime == "strong_multifractal":
            return (
                f"GÜÇLÜ ÇOK-FRAKTAL: Δh={p.delta_h:.3f} (yüksek heterojenlik). "
                f"Piyasa kaotik ve öngörülemez. Stake %50 düşür."
            )
        if r.regime == "weak_multifractal":
            return (
                f"ZAYIF ÇOK-FRAKTAL: Δh={p.delta_h:.3f}. "
                f"Dikkatli devam, küçük anomaliler mevcut."
            )
        trend = "TREND" if r.is_trending else ("ORTALAMAYA DÖNÜŞ" if r.is_mean_reverting else "RASTGELE")
        return (
            f"MONO-FRAKTAL ({trend}): h(2)={p.hurst_q2:.3f}, Δh={p.delta_h:.3f}. "
            f"Piyasa homojen ve tahmin edilebilir. Normal işlem."
        )
