"""
sde_pricer.py – Stochastic Differential Equations (SDEs).

Bahis oranları rastgele yürümez, bir "Mıknatıs" gibi Adil Değer'e
(Fair Value) çekilir. Bu hareketi Ornstein-Uhlenbeck (OU) süreci
ile modelleyeceğiz.

Ornstein-Uhlenbeck SDE:
  dX_t = θ(μ - X_t)dt + σ dW_t

  θ (theta): Mean-reversion hızı (ne kadar hızlı çekilir)
  μ (mu): Uzun vadeli ortalama (Adil Değer)
  σ (sigma): Volatilite (gürültü)
  W_t: Wiener süreci (Brownian motion)

Analitik Çözüm:
  X_t = μ + (X_0 - μ)e^{-θt} + σ√(1-e^{-2θt}/2θ) · Z

Sinyaller:
  - Oran μ'den uzaksa → Oran μ'ye doğru hareket edecek
  - σ yüksekse → Oran çok oynuyor, risk yüksek
  - θ yüksekse → Hızlı düzeltme, fırsatlar kısa ömürlü
  - X_t < μ ise → Oran yükselecek (bahis ucuzluyor)

Teknoloji: scipy + numpy (Euler-Maruyama, Milstein)
Fallback: Monte Carlo SDE simülasyonu
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from loguru import logger

try:
    from scipy.optimize import minimize
    from scipy.stats import norm
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False


# ═══════════════════════════════════════════════
#  VERİ YAPILARI
# ═══════════════════════════════════════════════
@dataclass
class OUParameters:
    """Ornstein-Uhlenbeck parametreleri."""
    theta: float = 0.0       # Mean-reversion hızı
    mu: float = 0.0          # Uzun vadeli ortalama (Fair Value)
    sigma: float = 0.0       # Volatilite
    half_life: float = 0.0   # Yarı ömür (ln(2)/θ)
    r_squared: float = 0.0   # Uyum kalitesi


@dataclass
class SDEForecast:
    """SDE tahmin raporu."""
    match_id: str = ""
    current_odds: float = 0.0
    fair_value: float = 0.0          # μ (adil değer)
    # Tahmin
    predicted_odds: float = 0.0      # t dakika sonra beklenen oran
    prediction_horizon_min: int = 10 # Tahmin ufku (dakika)
    # Yön
    expected_direction: str = ""     # "up" | "down" | "stable"
    expected_change_pct: float = 0.0
    # Güven aralığı
    ci_lower: float = 0.0           # %95 alt sınır
    ci_upper: float = 0.0           # %95 üst sınır
    # Parametreler
    params: OUParameters = field(default_factory=OUParameters)
    # Risk
    volatility_rank: str = ""       # "low" | "medium" | "high"
    opportunity_window_min: float = 0.0  # Fırsat penceresi
    # Sinyal
    value_signal: str = ""          # "BUY" | "SELL" | "HOLD"
    edge_pct: float = 0.0          # Value edge
    recommendation: str = ""
    method: str = ""


# ═══════════════════════════════════════════════
#  OU PARAMETRE TAHMİNİ (MLE)
# ═══════════════════════════════════════════════
def estimate_ou_params(prices: np.ndarray,
                        dt: float = 1.0) -> OUParameters:
    """Ornstein-Uhlenbeck parametrelerini MLE ile tahmin et.

    prices: Oran zaman serisi
    dt: Zaman adımı (dakika)
    """
    params = OUParameters()
    prices = np.array(prices, dtype=np.float64)
    n = len(prices)

    if n < 5:
        params.mu = float(np.mean(prices)) if n > 0 else 0
        return params

    # AR(1) regresyonu: X_{t+1} = a + b·X_t + ε
    X = prices[:-1]
    Y = prices[1:]

    # OLS
    n_obs = len(X)
    x_mean = np.mean(X)
    y_mean = np.mean(Y)

    Sxx = np.sum((X - x_mean) ** 2)
    Sxy = np.sum((X - x_mean) * (Y - y_mean))

    if abs(Sxx) < 1e-15:
        params.mu = float(np.mean(prices))
        return params

    b = Sxy / Sxx
    a = y_mean - b * x_mean

    # OU parametreleri
    if 0 < b < 1:
        params.theta = -np.log(b) / dt
        params.mu = a / (1 - b)
    elif b >= 1:
        # Trend (ortalamaya dönüş yok)
        params.theta = 0.01
        params.mu = float(np.mean(prices))
    else:
        params.theta = 2.0  # Çok hızlı dönüş
        params.mu = a / (1 - b) if abs(1 - b) > 1e-10 else float(np.mean(prices))

    # Volatilite
    residuals = Y - (a + b * X)
    sigma_residual = float(np.std(residuals))
    if params.theta > 0:
        params.sigma = sigma_residual * np.sqrt(
            2 * params.theta / (1 - np.exp(-2 * params.theta * dt))
        )
    else:
        params.sigma = sigma_residual

    # Yarı ömür
    if params.theta > 0:
        params.half_life = np.log(2) / params.theta
    else:
        params.half_life = float("inf")

    # R²
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((Y - y_mean) ** 2)
    params.r_squared = round(1 - ss_res / max(ss_tot, 1e-15), 4)

    # Clip
    params.theta = round(max(0.001, min(params.theta, 10.0)), 6)
    params.mu = round(params.mu, 4)
    params.sigma = round(max(0.001, params.sigma), 6)
    params.half_life = round(params.half_life, 2)

    return params


# ═══════════════════════════════════════════════
#  OU SÜREÇ SİMÜLASYONU
# ═══════════════════════════════════════════════
def ou_expected_value(x0: float, theta: float, mu: float,
                       t: float) -> float:
    """OU sürecinin beklenen değeri (analitik).

    E[X_t] = μ + (X_0 - μ) · e^{-θt}
    """
    return mu + (x0 - mu) * np.exp(-theta * t)


def ou_variance(theta: float, sigma: float, t: float) -> float:
    """OU sürecinin varyansı (analitik).

    Var[X_t] = σ²/(2θ) · (1 - e^{-2θt})
    """
    if theta <= 0:
        return sigma ** 2 * t
    return (sigma ** 2) / (2 * theta) * (1 - np.exp(-2 * theta * t))


def ou_confidence_interval(x0: float, theta: float, mu: float,
                             sigma: float, t: float,
                             alpha: float = 0.05
                             ) -> tuple[float, float, float]:
    """OU tahmin ve güven aralığı.

    Returns: (expected, ci_lower, ci_upper)
    """
    expected = ou_expected_value(x0, theta, mu, t)
    var = ou_variance(theta, sigma, t)
    std = np.sqrt(max(var, 0))

    if SCIPY_OK:
        z = norm.ppf(1 - alpha / 2)
    else:
        z = 1.96  # ~%95

    return expected, expected - z * std, expected + z * std


def simulate_ou_paths(x0: float, theta: float, mu: float,
                        sigma: float, T: float, dt: float = 1.0,
                        n_paths: int = 1000) -> np.ndarray:
    """OU süreci Monte Carlo simülasyonu (Euler-Maruyama).

    Returns: (n_steps, n_paths)
    """
    n_steps = int(T / dt) + 1
    paths = np.zeros((n_steps, n_paths))
    paths[0, :] = x0

    sqrt_dt = np.sqrt(dt)

    for t in range(1, n_steps):
        dW = np.random.randn(n_paths) * sqrt_dt
        drift = theta * (mu - paths[t - 1, :]) * dt
        diffusion = sigma * dW
        paths[t, :] = paths[t - 1, :] + drift + diffusion

    return paths


# ═══════════════════════════════════════════════
#  SDE PRICER (Ana Sınıf)
# ═══════════════════════════════════════════════
class SDEPricer:
    """Stokastik diferansiyel denklem ile oran fiyatlayıcı.

    Kullanım:
        pricer = SDEPricer()

        # Oran geçmişi (son 30 dakika)
        odds_history = [1.85, 1.83, 1.80, 1.82, 1.78, ...]

        # 10 dakika sonrasını tahmin et
        forecast = pricer.forecast(
            odds_history, match_id="gs_fb",
            horizon_min=10,
        )

        if forecast.value_signal == "BUY":
            bet_now()  # Oran yükselecek
    """

    # Volatilite sınıflandırma eşikleri
    LOW_VOL = 0.05
    HIGH_VOL = 0.20

    def __init__(self, dt: float = 1.0, n_sim_paths: int = 1000,
                 min_edge: float = 0.02):
        self._dt = dt                # Zaman adımı (dakika)
        self._n_paths = n_sim_paths
        self._min_edge = min_edge
        logger.debug(f"[SDE] Pricer başlatıldı (dt={dt}, paths={n_sim_paths})")

    def forecast(self, odds_history: list[float] | np.ndarray,
                   match_id: str = "",
                   horizon_min: int = 10,
                   fair_value_override: float | None = None
                   ) -> SDEForecast:
        """Oran tahmin raporu."""
        fc = SDEForecast(match_id=match_id, prediction_horizon_min=horizon_min)
        history = np.array(odds_history, dtype=np.float64)

        if len(history) < 3:
            fc.recommendation = "Yetersiz oran geçmişi (min 3)."
            return fc

        fc.current_odds = round(float(history[-1]), 4)

        # OU parametrelerini tahmin et
        params = estimate_ou_params(history, dt=self._dt)
        fc.params = params
        fc.fair_value = fair_value_override if fair_value_override else params.mu

        # Analitik tahmin
        expected, ci_lo, ci_hi = ou_confidence_interval(
            fc.current_odds, params.theta, fc.fair_value,
            params.sigma, horizon_min,
        )
        fc.predicted_odds = round(expected, 4)
        fc.ci_lower = round(ci_lo, 4)
        fc.ci_upper = round(ci_hi, 4)

        # Yön
        change = fc.predicted_odds - fc.current_odds
        fc.expected_change_pct = round(
            change / max(fc.current_odds, 1e-6) * 100, 3,
        )
        if abs(fc.expected_change_pct) < 0.5:
            fc.expected_direction = "stable"
        elif change > 0:
            fc.expected_direction = "up"
        else:
            fc.expected_direction = "down"

        # Volatilite sınıfı
        if params.sigma < self.LOW_VOL:
            fc.volatility_rank = "low"
        elif params.sigma > self.HIGH_VOL:
            fc.volatility_rank = "high"
        else:
            fc.volatility_rank = "medium"

        # Fırsat penceresi (yarı ömür)
        fc.opportunity_window_min = params.half_life

        # Value sinyali
        edge = (fc.fair_value - fc.current_odds) / max(fc.current_odds, 1e-6)
        fc.edge_pct = round(edge * 100, 3)

        if edge > self._min_edge:
            fc.value_signal = "BUY"   # Oran düşük, yükselecek
        elif edge < -self._min_edge:
            fc.value_signal = "SELL"  # Oran yüksek, düşecek
        else:
            fc.value_signal = "HOLD"

        fc.method = "ou_analytic"
        fc.recommendation = self._advice(fc)
        return fc

    def simulate(self, current_odds: float, theta: float,
                   mu: float, sigma: float,
                   horizon_min: int = 30) -> dict:
        """Monte Carlo SDE simülasyonu."""
        paths = simulate_ou_paths(
            current_odds, theta, mu, sigma,
            T=horizon_min, dt=self._dt,
            n_paths=self._n_paths,
        )

        final = paths[-1, :]
        return {
            "mean": round(float(np.mean(final)), 4),
            "median": round(float(np.median(final)), 4),
            "std": round(float(np.std(final)), 4),
            "p5": round(float(np.percentile(final, 5)), 4),
            "p95": round(float(np.percentile(final, 95)), 4),
            "prob_up": round(float(np.mean(final > current_odds)), 4),
            "prob_down": round(float(np.mean(final < current_odds)), 4),
            "n_paths": self._n_paths,
        }

    def batch_forecast(self, matches: list[dict],
                         odds_field: str = "odds_history",
                         horizon: int = 10) -> list[SDEForecast]:
        """Toplu maç tahmini."""
        results = []
        for match in matches:
            if not isinstance(match, dict):
                continue
            history = match.get(odds_field, [])
            if len(history) >= 3:
                fc = self.forecast(
                    history,
                    match_id=match.get("match_id", ""),
                    horizon_min=horizon,
                    fair_value_override=match.get("fair_odds"),
                )
                results.append(fc)
        return results

    def _advice(self, fc: SDEForecast) -> str:
        if fc.value_signal == "BUY":
            return (
                f"BUY: Oran {fc.current_odds} → beklenen {fc.predicted_odds} "
                f"({fc.expected_change_pct:+.1f}%, {fc.prediction_horizon_min}dk). "
                f"Fair={fc.fair_value:.2f}, edge={fc.edge_pct:+.1f}%. "
                f"Fırsat penceresi: {fc.opportunity_window_min:.0f}dk."
            )
        if fc.value_signal == "SELL":
            return (
                f"SELL: Oran {fc.current_odds} aşırı yüksek. "
                f"Fair={fc.fair_value:.2f}, {fc.prediction_horizon_min}dk sonra "
                f"beklenen={fc.predicted_odds} ({fc.expected_change_pct:+.1f}%)."
            )
        return (
            f"HOLD: Oran {fc.current_odds} ≈ fair ({fc.fair_value:.2f}). "
            f"Volatilite={fc.volatility_rank}, θ={fc.params.theta:.3f}."
        )
