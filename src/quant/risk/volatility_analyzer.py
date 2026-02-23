"""
volatility_analyzer.py – GARCH Volatility Clustering.

Finans dünyasında riskler sürüler halinde gelir. Bir ligde
sürprizler başladığında arkası kesilmez. GARCH bu "risk
dalgalarını" (volatility clustering) tespit eder.

Kavramlar:
  - GARCH(1,1): σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}
    → Koşullu varyans modeli
  - Volatility Clustering: Yüksek oynaklık dönemleri kümeleşir
    (sakin dönemler de kümeleşir)
  - ARCH Effect: Büyük şokların ardından daha büyük şoklar gelir
  - Conditional Volatility: σ_t — zamana bağlı oynaklık tahmini
  - VaR (Value at Risk): %α güvenle en kötü kayıp
  - Risk Regime: "calm" | "elevated" | "storm" | "crisis"
  - EWMA Volatility: Exponentially Weighted Moving Average (fallback)

Akış:
  1. Oran return serisini al (log-returns)
  2. GARCH(1,1) parametrelerini tahmin et (MLE)
  3. Koşullu oynaklık serisini çıkar σ_t
  4. Risk rejimi sınıflandır (σ_t eşik değerleri)
  5. Kelly çarpanını otomatik ayarla

Teknoloji: arch (Python ARCH kütüphanesi)
Fallback: EWMA + basit rolling volatility
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from loguru import logger

try:
    from arch import arch_model
    ARCH_OK = True
except ImportError:
    ARCH_OK = False
    logger.debug("arch yüklü değil – EWMA volatility fallback.")

try:
    from scipy.stats import norm as sp_norm
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False


# ═══════════════════════════════════════════════
#  VERİ YAPILARI
# ═══════════════════════════════════════════════
@dataclass
class GARCHParams:
    """GARCH(1,1) model parametreleri."""
    omega: float = 0.0       # Sabit terim (uzun vadeli varyans ağırlığı)
    alpha: float = 0.0       # ARCH katsayısı (şok etkisi)
    beta: float = 0.0        # GARCH katsayısı (kalıcılık)
    persistence: float = 0.0  # α + β (1'e yakınsa çok kalıcı)
    half_life: float = 0.0   # Yarı ömür (şokun etkisinin yarıya düşme süresi)
    long_run_var: float = 0.0  # ω / (1 - α - β) (uzun vadeli varyans)


@dataclass
class VolatilityReport:
    """Oynaklık analiz raporu."""
    match_id: str = ""
    market: str = "odds"
    team: str = ""
    # GARCH
    params: GARCHParams = field(default_factory=GARCHParams)
    # Mevcut durum
    current_volatility: float = 0.0   # σ_t (anlık oynaklık)
    avg_volatility: float = 0.0       # Ortalama σ
    volatility_percentile: float = 0.0  # Mevcut σ'nın tarihsel yüzdelik dilimi
    # Rejim
    regime: str = ""             # "calm" | "elevated" | "storm" | "crisis"
    regime_change: bool = False  # Rejim değişti mi?
    # Risk
    var_95: float = 0.0          # %95 VaR (1 günlük)
    var_99: float = 0.0          # %99 VaR
    kelly_multiplier: float = 1.0  # Kelly çarpanı (risk ayarı)
    # Forecast
    forecast_1d: float = 0.0     # 1 gün sonraki σ tahmini
    forecast_5d: float = 0.0     # 5 gün sonraki σ tahmini
    # Meta
    n_observations: int = 0
    method: str = ""
    recommendation: str = ""


# ═══════════════════════════════════════════════
#  EWMA FALLBACK
# ═══════════════════════════════════════════════
def ewma_volatility(returns: np.ndarray, span: int = 30) -> np.ndarray:
    """Exponentially Weighted Moving Average oynaklık."""
    alpha = 2.0 / (span + 1)
    n = len(returns)
    var = np.zeros(n)
    var[0] = returns[0] ** 2

    for i in range(1, n):
        var[i] = alpha * returns[i] ** 2 + (1 - alpha) * var[i - 1]

    return np.sqrt(var)


def rolling_volatility(returns: np.ndarray, window: int = 20) -> np.ndarray:
    """Basit kayan pencere oynaklık."""
    n = len(returns)
    vol = np.zeros(n)
    for i in range(n):
        start = max(0, i - window + 1)
        vol[i] = np.std(returns[start:i + 1]) if i > 0 else abs(returns[0])
    return vol


def garch_ewma_params(returns: np.ndarray, span: int = 30) -> GARCHParams:
    """EWMA'dan GARCH-benzeri parametre tahmini."""
    alpha = 2.0 / (span + 1)
    beta = 1 - alpha
    omega = np.var(returns) * alpha * 0.01

    persistence = alpha + beta
    half_life = np.log(2) / max(-np.log(beta), 1e-6) if beta < 1 else float("inf")
    lr_var = omega / max(1 - persistence, 1e-6) if persistence < 1 else np.var(returns)

    return GARCHParams(
        omega=round(float(omega), 8),
        alpha=round(float(alpha), 6),
        beta=round(float(beta), 6),
        persistence=round(float(persistence), 6),
        half_life=round(float(half_life), 2),
        long_run_var=round(float(lr_var), 8),
    )


# ═══════════════════════════════════════════════
#  VOLATILITY ANALYZER (Ana Sınıf)
# ═══════════════════════════════════════════════
class VolatilityAnalyzer:
    """GARCH tabanlı oynaklık analizi.

    Kullanım:
        va = VolatilityAnalyzer()

        # Analiz
        report = va.analyze(odds_returns, match_id="gs_fb", team="GS")

        # Kelly çarpanı
        kelly_mult = report.kelly_multiplier  # Örn: 0.6 → stake %40 düşür
    """

    # Rejim eşikleri (σ'nın uzun vadeli ortalamaya oranı)
    CALM_THRESHOLD = 0.8        # σ < 0.8 × avg → sakin
    ELEVATED_THRESHOLD = 1.3    # σ < 1.3 × avg → yükselmiş
    STORM_THRESHOLD = 2.0       # σ < 2.0 × avg → fırtına
    # Üstü → kriz

    # Kelly çarpanları (risk azaltma)
    KELLY_MAP = {
        "calm": 1.0,
        "elevated": 0.75,
        "storm": 0.40,
        "crisis": 0.0,
    }

    def __init__(self, model_type: str = "GARCH",
                 p: int = 1, q: int = 1,
                 ewma_span: int = 30):
        self._model_type = model_type
        self._p = p
        self._q = q
        self._ewma_span = ewma_span
        self._last_regime: str = ""
        self._history: list[VolatilityReport] = []

        logger.debug(
            f"[Volatility] Analyzer başlatıldı: "
            f"type={model_type}({p},{q}), "
            f"arch={'OK' if ARCH_OK else 'fallback'}"
        )

    def analyze(self, returns: np.ndarray | list,
                  match_id: str = "",
                  team: str = "",
                  market: str = "odds") -> VolatilityReport:
        """Oynaklık analizi."""
        report = VolatilityReport(match_id=match_id, team=team, market=market)
        data = np.array(returns, dtype=np.float64)
        data = data[np.isfinite(data)]
        report.n_observations = len(data)

        if len(data) < 20:
            report.regime = "insufficient_data"
            report.kelly_multiplier = 0.5
            report.recommendation = f"Yetersiz veri ({len(data)} gözlem, min 20)."
            report.method = "none"
            return report

        # Ölçekleme (yüzdeye çevir)
        data_scaled = data * 100 if np.max(np.abs(data)) < 1 else data

        if ARCH_OK:
            report = self._analyze_garch(data_scaled, report)
        else:
            report = self._analyze_ewma(data, report)

        # Rejim sınıflandırma
        report = self._classify_regime(report)

        # VaR hesaplama
        report = self._compute_var(data, report)

        # Forecast
        report = self._forecast(report)

        # Rejim değişimi
        if self._last_regime and self._last_regime != report.regime:
            report.regime_change = True
        self._last_regime = report.regime

        report.recommendation = self._advice(report)
        self._history.append(report)
        return report

    def _analyze_garch(self, data: np.ndarray,
                         report: VolatilityReport) -> VolatilityReport:
        """arch kütüphanesi ile GARCH fit."""
        try:
            model = arch_model(
                data,
                vol=self._model_type,
                p=self._p, q=self._q,
                dist="normal",
                rescale=True,
            )
            result = model.fit(disp="off", show_warning=False)

            # Parametreler
            params = result.params
            report.params = GARCHParams(
                omega=round(float(params.get("omega", 0)), 8),
                alpha=round(float(params.get("alpha[1]", 0)), 6),
                beta=round(float(params.get("beta[1]", 0)), 6),
            )
            p = report.params
            p.persistence = round(p.alpha + p.beta, 6)
            if p.persistence < 1 and p.beta > 0:
                p.half_life = round(
                    float(np.log(2) / max(-np.log(p.beta), 1e-6)), 2,
                )
            if 1 - p.persistence > 1e-6:
                p.long_run_var = round(
                    float(p.omega / (1 - p.persistence)), 8,
                )

            # Koşullu oynaklık serisi
            cond_vol = result.conditional_volatility
            if len(cond_vol) > 0:
                report.current_volatility = round(
                    float(cond_vol.iloc[-1]) / 100, 6,
                )
                report.avg_volatility = round(
                    float(np.mean(cond_vol)) / 100, 6,
                )
                report.volatility_percentile = round(
                    float(np.mean(cond_vol <= cond_vol.iloc[-1])) * 100, 1,
                )

            report.method = f"garch({self._p},{self._q})"

        except Exception as e:
            logger.debug(f"[Volatility] GARCH hatası: {e}")
            data_orig = data / 100 if np.max(np.abs(data)) > 10 else data
            report = self._analyze_ewma(data_orig, report)

        return report

    def _analyze_ewma(self, data: np.ndarray,
                        report: VolatilityReport) -> VolatilityReport:
        """EWMA fallback."""
        vol_series = ewma_volatility(data, span=self._ewma_span)

        report.current_volatility = round(float(vol_series[-1]), 6)
        report.avg_volatility = round(float(np.mean(vol_series)), 6)
        report.volatility_percentile = round(
            float(np.mean(vol_series <= vol_series[-1])) * 100, 1,
        )

        report.params = garch_ewma_params(data, self._ewma_span)
        report.method = f"ewma(span={self._ewma_span})"
        return report

    def _classify_regime(self, report: VolatilityReport) -> VolatilityReport:
        """Risk rejimi sınıflandırma."""
        avg = report.avg_volatility
        cur = report.current_volatility

        if avg <= 0:
            report.regime = "calm"
            report.kelly_multiplier = 1.0
            return report

        ratio = cur / avg

        if ratio < self.CALM_THRESHOLD:
            report.regime = "calm"
        elif ratio < self.ELEVATED_THRESHOLD:
            report.regime = "elevated"
        elif ratio < self.STORM_THRESHOLD:
            report.regime = "storm"
        else:
            report.regime = "crisis"

        report.kelly_multiplier = self.KELLY_MAP.get(report.regime, 0.5)
        return report

    def _compute_var(self, data: np.ndarray,
                       report: VolatilityReport) -> VolatilityReport:
        """Value at Risk hesaplama."""
        sigma = report.current_volatility
        if sigma <= 0:
            return report

        if SCIPY_OK:
            report.var_95 = round(float(sp_norm.ppf(0.05) * sigma), 6)
            report.var_99 = round(float(sp_norm.ppf(0.01) * sigma), 6)
        else:
            report.var_95 = round(-1.645 * sigma, 6)
            report.var_99 = round(-2.326 * sigma, 6)

        return report

    def _forecast(self, report: VolatilityReport) -> VolatilityReport:
        """Oynaklık tahmini (1 gün, 5 gün)."""
        p = report.params
        cur_var = report.current_volatility ** 2

        if p.persistence > 0 and p.persistence < 1:
            lr_var = p.long_run_var if p.long_run_var > 0 else cur_var

            # h-adım tahmini: σ²_h = LR + (α+β)^h × (σ²_0 - LR)
            var_1 = lr_var + p.persistence * (cur_var - lr_var)
            var_5 = lr_var + (p.persistence ** 5) * (cur_var - lr_var)

            report.forecast_1d = round(float(np.sqrt(max(var_1, 0))), 6)
            report.forecast_5d = round(float(np.sqrt(max(var_5, 0))), 6)
        else:
            report.forecast_1d = report.current_volatility
            report.forecast_5d = report.current_volatility

        return report

    def analyze_league(self, league_returns: dict[str, np.ndarray],
                         league_name: str = "") -> dict[str, VolatilityReport]:
        """Tüm lig takımlarının oynaklık analizi."""
        results = {}
        for team, returns in league_returns.items():
            results[team] = self.analyze(
                returns, team=team, market="league",
            )
        return results

    def _advice(self, r: VolatilityReport) -> str:
        p = r.params
        if r.regime == "crisis":
            return (
                f"KRİZ: {r.team} oynaklık krizde! "
                f"σ={r.current_volatility:.4f} "
                f"(ort.nın {r.volatility_percentile:.0f}. yüzdeliği). "
                f"Kelly x{r.kelly_multiplier:.1f}. "
                f"TÜM BAHİSLERİ DURDUR."
            )
        if r.regime == "storm":
            return (
                f"FIRTINA: σ={r.current_volatility:.4f}, "
                f"persistence={p.persistence:.2f}. "
                f"Kelly x{r.kelly_multiplier:.1f}. "
                f"Stake %60 düşür."
            )
        if r.regime == "elevated":
            return (
                f"YÜKSELMİŞ: σ={r.current_volatility:.4f}. "
                f"Kelly x{r.kelly_multiplier:.1f}. "
                f"Dikkatli devam."
                + (f" Rejim değişti!" if r.regime_change else "")
            )
        return (
            f"SAKİN: σ={r.current_volatility:.4f}, "
            f"VaR95={r.var_95:.4f}. "
            f"Kelly x{r.kelly_multiplier:.1f}. Normal işlem."
        )
