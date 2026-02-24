"""
fractal_analyzer.py – Hurst Exponent ve Fraktal Geometri Analizi.

Piyasalar rastgele yürümez, "Hafızası" vardır. Bir ligin "Trend"
mi yoksa "Ortalamaya Dönüş" (Mean Reverting) modunda mı olduğunu
Fraktal Geometri ile ölçeriz.

Hurst Exponent (H):
  H < 0.5 (Mean Reverting): Lig çok sürprizli.
    Favori yenilirse bir sonraki maç kesin kazanır.
    Strateji: Anti-Trend / Contrarian

  H ≈ 0.5 (Random Walk): Tamamen rastgele.
    Hiçbir kalıp yok → Bahisten kaçın.

  H > 0.5 (Trending): Lig seriye bağlıyor.
    Kazanan kazanmaya devam eder.
    Strateji: Trend Follower / Momentum

Kavramlar:
  - R/S Analysis: Rescaled Range analizi (klasik Hurst)
  - DFA (Detrended Fluctuation Analysis): Trend çıkarılmış dalgalanma
  - Fractal Dimension: D = 2 - H (karmaşıklık ölçüsü)
  - Self-Similarity: Kendini tekrar eden kalıplar
  - Long-Range Dependence: Uzun vadeli hafıza

Sinyal:
  H 0.5'ten uzaksa (0.3 veya 0.8) → "Kalıp var" → Bahis artır
  H ≈ 0.5 → "Rastgele yürüyüş" → Bahisten kaçın
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from loguru import logger

try:
    from hurst import compute_Hc
    HURST_OK = True
except ImportError:
    HURST_OK = False
    logger.info("hurst yüklü değil – R/S analiz fallback.")

try:
    from scipy import stats as sp_stats
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False


@dataclass
class HurstResult:
    """Hurst Exponent sonucu."""
    hurst: float = 0.5
    confidence: float = 0.0          # R² uyum güvenilirliği
    regime: str = "random"           # trending | mean_reverting | random
    fractal_dimension: float = 1.5   # D = 2 - H
    # Detay
    method: str = ""
    series_length: int = 0
    # Strateji sinyali
    pattern_strength: float = 0.0    # |H - 0.5| (0=rastgele, 0.5=güçlü kalıp)
    recommended_strategy: str = ""
    kelly_multiplier: float = 1.0    # H'ye göre bahis çarpanı


@dataclass
class FractalReport:
    """Kapsamlı fraktal analiz raporu."""
    entity: str = ""                 # Takım veya lig ismi
    entity_type: str = ""            # team | league
    # Hurst
    hurst_result: HurstResult = field(default_factory=HurstResult)
    # DFA
    dfa_alpha: float = 0.0           # DFA scaling exponent
    dfa_regime: str = ""
    # Otokorelasyon
    autocorrelation_lag1: float = 0.0
    autocorrelation_lag5: float = 0.0
    # Kararlılık
    is_stable: bool = True           # Hurst kararlı mı?
    rolling_hurst: list[float] = field(default_factory=list)
    # Tavsiye
    recommendation: str = ""
    telegram_text: str = ""


# ═══════════════════════════════════════════════
#  HURST EXPONENT HESAPLAYICILARI
# ═══════════════════════════════════════════════
def hurst_rs(data: np.ndarray) -> tuple[float, float]:
    """Rescaled Range (R/S) analizi ile Hurst Exponent.

    Klasik yöntem: logaritmik R/S vs log(n) regresyonu.

    Returns: (hurst_value, r_squared)
    """
    n = len(data)
    if n < 20:
        return 0.5, 0.0

    # R/S hesaplama
    max_k = min(n // 2, 200)
    sizes = []
    rs_values = []

    for k in range(10, max_k, max(1, max_k // 20)):
        rs_list = []
        for start in range(0, n - k, k):
            segment = data[start:start + k]
            mean_seg = np.mean(segment)
            deviations = segment - mean_seg
            cumulative = np.cumsum(deviations)
            r = np.max(cumulative) - np.min(cumulative)
            s = np.std(segment, ddof=1)
            if s > 1e-10:
                rs_list.append(r / s)

        if rs_list:
            sizes.append(k)
            rs_values.append(np.mean(rs_list))

    if len(sizes) < 3:
        return 0.5, 0.0

    # Log-log regresyon
    log_sizes = np.log(sizes)
    log_rs = np.log(np.maximum(rs_values, 1e-10))

    if SCIPY_OK:
        slope, intercept, r_value, p_value, std_err = sp_stats.linregress(
            log_sizes, log_rs,
        )
        return float(np.clip(slope, 0.01, 0.99)), float(r_value ** 2)
    else:
        # Manuel regresyon
        n_pts = len(log_sizes)
        sum_x = np.sum(log_sizes)
        sum_y = np.sum(log_rs)
        sum_xy = np.sum(log_sizes * log_rs)
        sum_x2 = np.sum(log_sizes ** 2)
        denom = n_pts * sum_x2 - sum_x ** 2
        if abs(denom) < 1e-10:
            return 0.5, 0.0
        slope = (n_pts * sum_xy - sum_x * sum_y) / denom
        return float(np.clip(slope, 0.01, 0.99)), 0.5


def hurst_dfa(data: np.ndarray) -> float:
    """Detrended Fluctuation Analysis (DFA).

    Daha modern ve gürbüz bir Hurst tahmincisi.
    """
    n = len(data)
    if n < 20:
        return 0.5

    # Kümülatif toplam (profil)
    profile = np.cumsum(data - np.mean(data))

    # Farklı pencere boyutları
    min_win = 4
    max_win = n // 4
    if max_win <= min_win:
        return 0.5

    windows = np.unique(np.logspace(
        np.log10(min_win), np.log10(max_win), 15, dtype=int,
    ))
    windows = windows[windows >= min_win]

    fluctuations = []
    valid_windows = []

    for w in windows:
        segments = n // w
        if segments < 1:
            continue

        f_sq = []
        for seg in range(segments):
            start = seg * w
            end = start + w
            segment = profile[start:end]

            # Lokal trend (lineer)
            x = np.arange(w)
            if SCIPY_OK:
                slope, intercept, _, _, _ = sp_stats.linregress(x, segment)
                trend = intercept + slope * x
            else:
                coeffs = np.polyfit(x, segment, 1)
                trend = np.polyval(coeffs, x)

            residual = segment - trend
            f_sq.append(np.mean(residual ** 2))

        if f_sq:
            fluctuations.append(np.sqrt(np.mean(f_sq)))
            valid_windows.append(w)

    if len(valid_windows) < 3:
        return 0.5

    # Log-log regresyon
    log_w = np.log(valid_windows)
    log_f = np.log(np.maximum(fluctuations, 1e-10))

    if SCIPY_OK:
        slope, _, _, _, _ = sp_stats.linregress(log_w, log_f)
    else:
        coeffs = np.polyfit(log_w, log_f, 1)
        slope = coeffs[0]

    return float(np.clip(slope, 0.01, 0.99))


# ═══════════════════════════════════════════════
#  FRAKTAL ANALİZÖR
# ═══════════════════════════════════════════════
class FractalAnalyzer:
    """Hurst Exponent ve fraktal geometri ile piyasa analizi.

    Kullanım:
        analyzer = FractalAnalyzer()

        # Takım performans serisi (son 50 maç xG)
        data = [1.2, 0.8, 1.5, ...]
        result = analyzer.compute_hurst(data)

        # Lig analizi
        report = analyzer.analyze_entity(
            data=data,
            entity="Süper Lig",
            entity_type="league",
        )

        # Bahis stratejisi
        multiplier = analyzer.get_kelly_multiplier(data)
    """

    # Rejim eşikleri
    MEAN_REVERT_THRESHOLD = 0.40
    RANDOM_LOW = 0.45
    RANDOM_HIGH = 0.55
    TREND_THRESHOLD = 0.60

    def __init__(self):
        logger.debug("[Fractal] Analyzer başlatıldı.")

    def compute_hurst(self, data: list[float] | np.ndarray,
                       method: str = "auto") -> HurstResult:
        """Hurst Exponent hesapla.

        Args:
            method: "rs" | "dfa" | "hurst_lib" | "auto"
        """
        arr = np.array(data, dtype=np.float64)
        result = HurstResult(series_length=len(arr))

        if len(arr) < 20:
            result.method = "insufficient"
            result.recommended_strategy = "Yetersiz veri – bahisten kaçın."
            return result

        # ── Yöntem seçimi ──
        if method == "auto":
            if HURST_OK:
                method = "hurst_lib"
            else:
                method = "rs"

        if method == "hurst_lib" and HURST_OK:
            try:
                h, c, data_rs = compute_Hc(arr, kind="price", simplified=True)
                result.hurst = round(float(h), 4)
                result.confidence = round(float(c), 4) if c else 0.5
                result.method = "hurst_library"
            except Exception:
                method = "rs"

        if method == "rs":
            h, r2 = hurst_rs(arr)
            result.hurst = round(h, 4)
            result.confidence = round(r2, 4)
            result.method = "rescaled_range"

        elif method == "dfa":
            h = hurst_dfa(arr)
            result.hurst = round(h, 4)
            result.confidence = 0.7  # DFA genelde güvenilir
            result.method = "dfa"

        # ── Rejim ──
        h = result.hurst
        if h < self.MEAN_REVERT_THRESHOLD:
            result.regime = "mean_reverting"
        elif h < self.RANDOM_LOW:
            result.regime = "weak_mean_reverting"
        elif h <= self.RANDOM_HIGH:
            result.regime = "random"
        elif h <= self.TREND_THRESHOLD:
            result.regime = "weak_trending"
        else:
            result.regime = "trending"

        # ── Fraktal Boyut ──
        result.fractal_dimension = round(2 - h, 4)

        # ── Kalıp Gücü ──
        result.pattern_strength = round(abs(h - 0.5), 4)

        # ── Strateji ──
        result.recommended_strategy = self._recommend_strategy(result)

        # ── Kelly Çarpanı ──
        result.kelly_multiplier = self._kelly_multiplier(h)

        return result

    def _recommend_strategy(self, result: HurstResult) -> str:
        """Hurst'e göre strateji tavsiyesi."""
        h = result.hurst
        regime = result.regime

        if regime == "mean_reverting":
            return (
                f"CONTRARIAN (Anti-Trend): H={h:.2f}. "
                f"Favori yenilirse → Sonraki maç koyun. "
                f"Sürpriz beklentisi yüksek."
            )
        elif regime == "weak_mean_reverting":
            return (
                f"Zayıf Ortalamaya Dönüş: H={h:.2f}. "
                f"Dikkatli contrarian strateji."
            )
        elif regime == "random":
            return (
                f"RASTGELE YÜRÜYÜŞ: H={h:.2f}. "
                f"Kalıp YOK → Stake düşür veya pas geç."
            )
        elif regime == "weak_trending":
            return (
                f"Zayıf Trend: H={h:.2f}. "
                f"Temkinli momentum stratejisi."
            )
        else:
            return (
                f"TREND FOLLOWER (Momentum): H={h:.2f}. "
                f"Kazanan kazanmaya devam eder. "
                f"Form takımlarına koyun."
            )

    def _kelly_multiplier(self, h: float) -> float:
        """Hurst'e göre Kelly çarpanı.

        Kalıp güçlüyse → bahis artır
        Rastgele → bahis düşür
        """
        pattern = abs(h - 0.5)
        if pattern < 0.05:
            return 0.5   # Rastgele → %50 düşür
        elif pattern < 0.10:
            return 0.75
        elif pattern < 0.20:
            return 1.0
        else:
            return 1.25  # Güçlü kalıp → %25 artır

    # ═══════════════════════════════════════════
    #  KAPSAMLI ANALİZ
    # ═══════════════════════════════════════════
    def analyze_entity(self, data: list[float] | np.ndarray,
                        entity: str = "",
                        entity_type: str = "team"
                        ) -> FractalReport:
        """Takım veya lig için kapsamlı fraktal analiz."""
        arr = np.array(data, dtype=np.float64)
        report = FractalReport(entity=entity, entity_type=entity_type)

        # Hurst
        report.hurst_result = self.compute_hurst(arr)

        # DFA (ayrıca)
        report.dfa_alpha = round(hurst_dfa(arr), 4)
        if report.dfa_alpha < 0.5:
            report.dfa_regime = "anti-persistent"
        elif report.dfa_alpha > 0.5:
            report.dfa_regime = "persistent"
        else:
            report.dfa_regime = "random"

        # Otokorelasyon
        if len(arr) > 5:
            mean_arr = np.mean(arr)
            var_arr = np.var(arr)
            if var_arr > 1e-10:
                for lag, attr in [(1, "autocorrelation_lag1"),
                                   (5, "autocorrelation_lag5")]:
                    if len(arr) > lag:
                        ac = np.mean(
                            (arr[:-lag] - mean_arr) * (arr[lag:] - mean_arr)
                        ) / var_arr
                        setattr(report, attr, round(float(ac), 4))

        # Rolling Hurst (kararlılık kontrolü)
        window = min(30, len(arr) // 3)
        if window >= 20:
            for i in range(0, len(arr) - window, max(1, window // 3)):
                segment = arr[i:i + window]
                h_seg = self.compute_hurst(segment, method="rs")
                report.rolling_hurst.append(round(h_seg.hurst, 4))

            if report.rolling_hurst:
                h_std = float(np.std(report.rolling_hurst))
                report.is_stable = h_std < 0.15

        # Tavsiye
        report.recommendation = self._entity_advice(report)
        report.telegram_text = self._format_telegram(report)

        return report

    def _entity_advice(self, report: FractalReport) -> str:
        """Kapsamlı tavsiye."""
        h = report.hurst_result
        parts = [h.recommended_strategy]

        if not report.is_stable:
            parts.append(
                "UYARI: Hurst kararsız – rejim değişimi olabilir."
            )

        if abs(report.autocorrelation_lag1) > 0.3:
            parts.append(
                f"Güçlü kısa vadeli hafıza (AC₁={report.autocorrelation_lag1:.2f})."
            )

        return " | ".join(parts)

    def _format_telegram(self, report: FractalReport) -> str:
        """Telegram HTML formatı."""
        h = report.hurst_result

        regime_emoji = {
            "trending": "📈", "weak_trending": "📊",
            "random": "🎲", "weak_mean_reverting": "🔄",
            "mean_reverting": "🔁",
        }
        emoji = regime_emoji.get(h.regime, "📉")

        bar_len = 20
        fill = int(bar_len * h.hurst)
        bar = "█" * fill + "░" * (bar_len - fill)

        return (
            f"{emoji} <b>Fraktal Analiz: {report.entity}</b>\n"
            f"{'━' * 30}\n\n"
            f"Hurst Exponent: {h.hurst:.3f}\n"
            f"[{bar}] H={'%.2f' % h.hurst}\n"
            f"Rejim: {h.regime.upper()}\n"
            f"Fraktal Boyut: {h.fractal_dimension:.3f}\n"
            f"Kalıp Gücü: {h.pattern_strength:.2f}\n"
            f"Kelly Çarpanı: {h.kelly_multiplier:.2f}x\n\n"
            f"DFA α: {report.dfa_alpha:.3f}\n"
            f"AC(1): {report.autocorrelation_lag1:.3f}\n"
            f"AC(5): {report.autocorrelation_lag5:.3f}\n\n"
            f"💡 {report.recommendation[:200]}"
        )

    # ═══════════════════════════════════════════
    #  TOPLU ANALİZ
    # ═══════════════════════════════════════════
    def analyze_league(self, match_results: list[dict],
                        league: str = "") -> FractalReport:
        """Lig geneli fraktal analiz.

        match_results: [{"home_goals": 2, "away_goals": 1}, ...]
        """
        total_goals = [
            r.get("home_goals", 0) + r.get("away_goals", 0)
            for r in match_results
        ]
        return self.analyze_entity(total_goals, entity=league,
                                    entity_type="league")

    def get_kelly_multiplier(self, data: list[float] | np.ndarray
                              ) -> float:
        """Hurst'e göre Kelly çarpanını hızlı hesapla."""
        result = self.compute_hurst(data, method="rs")
        return result.kelly_multiplier

    def filter_by_regime(self, bets: list[dict],
                          data_map: dict[str, list[float]]
                          ) -> list[dict]:
        """Rastgele yürüyen (H≈0.5) liglerden gelen bahisleri düşür."""
        filtered = []
        for bet in bets:
            league = bet.get("league", "")
            if league in data_map:
                result = self.compute_hurst(data_map[league], method="rs")
                bet["hurst"] = result.hurst
                bet["hurst_regime"] = result.regime
                bet["hurst_kelly_mult"] = result.kelly_multiplier

                if result.regime == "random":
                    logger.info(
                        f"[Fractal] {bet.get('match_id','')}: "
                        f"Lig '{league}' rastgele yürüyüş "
                        f"(H={result.hurst:.2f}) → stake düşürüldü"
                    )
                    # Rastgele rejimde stake düşür
                    original = bet.get("kelly_stake", bet.get("stake", 100))
                    bet["kelly_stake"] = round(
                        original * result.kelly_multiplier, 2,
                    )

            filtered.append(bet)
        return filtered
