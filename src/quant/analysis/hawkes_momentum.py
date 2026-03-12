"""
hawkes_momentum.py – Hawkes Processes (Self-Exciting Point Processes).

Goller bağımsız değildir. Bir gol, ikinci golün ihtimalini
artırır (Momentum/Panic). Poisson bunu göremez, Hawkes görür.

Kavramlar:
  - Self-Excitation: Bir olay gelecek olayların olasılığını artırır
  - Conditional Intensity: λ*(t) = μ + Σ α·e^{-β(t-tᵢ)}
    μ: Arka plan yoğunluğu (base rate)
    α: Uyarım büyüklüğü (excitement)
    β: Sönüm hızı (decay rate)
  - Branching Ratio: n = α/β (≈1 → kritik, >1 → patlama)
  - Decay Kernel: Olayın etkisinin zamanla azalması

Sinyaller:
  - Branching ratio < 0.5 → Maç sakin, olaylar bağımsız
  - Branching ratio ≈ 0.8 → Momentum artıyor
  - Branching ratio ≈ 1.0 → Patlama noktası (Criticality)!
  - Gol sonrası 5dk içinde 2. gol olasılığı → Canlı bahis sinyali

Teknoloji: tick kütüphanesi
Fallback: Manuel MLE (scipy.optimize)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from loguru import logger

try:
    from tick.hawkes import HawkesExpKern, SimuHawkesExpKernels
    TICK_OK = True
except ImportError:
    TICK_OK = False
    logger.debug("tick yüklü değil – manuel Hawkes fallback.")

try:
    from scipy.optimize import minimize
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False


# ═══════════════════════════════════════════════
#  VERİ YAPILARI
# ═══════════════════════════════════════════════
@dataclass
class HawkesParams:
    """Hawkes süreci parametreleri."""
    mu: float = 0.0          # Arka plan yoğunluğu (base rate)
    alpha: float = 0.0       # Uyarım büyüklüğü
    beta: float = 0.0        # Sönüm hızı
    branching_ratio: float = 0.0   # n = α/β
    half_life: float = 0.0   # Etki yarı ömrü (ln(2)/β dakika)


@dataclass
class HawkesReport:
    """Hawkes momentum raporu."""
    match_id: str = ""
    # Parametreler
    params: HawkesParams = field(default_factory=HawkesParams)
    # Mevcut durum
    current_intensity: float = 0.0     # Anlık yoğunluk
    baseline_intensity: float = 0.0    # Arka plan
    excitement_ratio: float = 0.0      # Uyarılma oranı
    # Olaylar
    n_events: int = 0
    last_event_min: float = 0.0        # Son olayın dakikası
    time_since_last: float = 0.0       # Son olaydan beri geçen süre
    # Tahmin
    next_event_prob_5min: float = 0.0  # 5dk içinde olay olasılığı
    next_event_prob_10min: float = 0.0
    # Rejim
    criticality: str = "subcritical"   # "subcritical" | "critical" | "supercritical"
    momentum_level: str = "low"        # "low" | "medium" | "high" | "explosive"
    # Sinyal
    over_signal: bool = False          # Üst bahis sinyali
    goal_burst_alert: bool = False     # Gol patlaması uyarısı
    recommendation: str = ""
    method: str = ""


# ═══════════════════════════════════════════════
#  HAWKES YOĞUNLUK HESAPLAMALARI
# ═══════════════════════════════════════════════
def hawkes_intensity(t: float, events: np.ndarray,
                      mu: float, alpha: float,
                      beta: float) -> float:
    """Hawkes koşullu yoğunluğu.

    λ*(t) = μ + Σ α · exp(-β(t - tᵢ))  [tᵢ < t]
    """
    past = events[events < t]
    if len(past) == 0:
        return mu
    excitement = alpha * np.sum(np.exp(-beta * (t - past)))
    return mu + excitement


def hawkes_log_likelihood(params: np.ndarray,
                            events: np.ndarray,
                            T: float) -> float:
    """Negatif log-likelihood (minimize edilecek).

    L = Σ log λ*(tᵢ) - ∫₀ᵀ λ*(t) dt
    """
    mu, alpha, beta = params
    if mu <= 0 or alpha < 0 or beta <= 0:
        return 1e10

    n = len(events)
    if n == 0:
        return mu * T

    # Σ log λ*(tᵢ)
    log_sum = 0.0
    for i in range(n):
        lam = hawkes_intensity(events[i], events, mu, alpha, beta)
        if lam > 0:
            log_sum += np.log(lam)
        else:
            return 1e10

    # ∫₀ᵀ λ*(t) dt = μT + (α/β) Σ (1 - exp(-β(T - tᵢ)))
    integral = mu * T
    for ti in events:
        integral += (alpha / beta) * (1 - np.exp(-beta * (T - ti)))

    return -(log_sum - integral)


def estimate_hawkes_params(events: np.ndarray,
                             T: float = 90.0) -> HawkesParams:
    """Hawkes parametrelerini MLE ile tahmin et."""
    params = HawkesParams()
    events = np.sort(events.flatten())
    n = len(events)

    if n < 2:
        params.mu = n / max(T, 1)
        return params

    if TICK_OK:
        try:
            learner = HawkesExpKern(1e-3, max_iter=1000)
            learner.fit([events.reshape(1, -1) if events.ndim == 1 else events])
            params.mu = float(learner.baseline[0])
            params.alpha = float(learner.adjacency[0, 0])
            params.beta = float(learner.decays[0, 0]) if hasattr(learner, "decays") else 1.0
        except Exception as e:
            logger.debug(f"Exception caught: {e}")
            params = _estimate_manual(events, T)
    elif SCIPY_OK:
        params = _estimate_manual(events, T)
    else:
        params.mu = n / max(T, 1)
        return params

    # Türetilmiş metrikler
    if params.beta > 0:
        params.branching_ratio = round(params.alpha / params.beta, 4)
        params.half_life = round(np.log(2) / params.beta, 2)
    else:
        params.branching_ratio = 0.0
        params.half_life = float("inf")

    return params


def _estimate_manual(events: np.ndarray, T: float) -> HawkesParams:
    """Manuel MLE (scipy.optimize)."""
    params = HawkesParams()
    n = len(events)

    # Başlangıç tahminleri
    mu0 = n / max(T, 1) * 0.5
    alpha0 = 0.5
    beta0 = 1.0

    try:
        result = minimize(
            hawkes_log_likelihood,
            x0=[mu0, alpha0, beta0],
            args=(events, T),
            method="L-BFGS-B",
            bounds=[(1e-6, 10), (0, 10), (1e-3, 50)],
        )
        if result.success:
            params.mu = round(float(result.x[0]), 6)
            params.alpha = round(float(result.x[1]), 6)
            params.beta = round(float(result.x[2]), 6)
        else:
            params.mu = round(mu0, 6)
    except Exception as e:
        logger.debug(f"Exception caught: {e}")
        params.mu = round(n / max(T, 1), 6)

    return params


# ═══════════════════════════════════════════════
#  HAWKES MOMENTUM ANALYZER (Ana Sınıf)
# ═══════════════════════════════════════════════
class HawkesMomentumAnalyzer:
    """Hawkes süreçleri ile momentum ve bulaşıcılık analizi.

    Kullanım:
        hma = HawkesMomentumAnalyzer()

        # Maç olayları (dakika cinsinden)
        goals = [12.0, 35.0, 37.0, 78.0]  # 37' → 35' golünün uyarısı!
        cards = [23.0, 55.0]

        report = hma.analyze_match(goals, match_id="gs_fb", current_min=80)
        if report.goal_burst_alert:
            place_over_bet()
    """

    CRITICAL_THRESHOLD = 0.85     # Branching ratio
    EXPLOSIVE_THRESHOLD = 0.95

    def __init__(self, match_duration: float = 90.0):
        self._T = match_duration
        logger.debug("[Hawkes] MomentumAnalyzer başlatıldı.")

    def analyze_match(self, event_times: list[float] | np.ndarray,
                        match_id: str = "",
                        current_min: float = 90.0,
                        event_type: str = "goal") -> HawkesReport:
        """Maç olaylarını Hawkes süreci ile analiz et."""
        report = HawkesReport(match_id=match_id)
        events = np.array(sorted(event_times), dtype=np.float64)
        report.n_events = len(events)

        if len(events) < 1:
            report.recommendation = "Olay yok – analiz yapılamadı."
            report.method = "no_data"
            return report

        # Parametre tahmini
        params = estimate_hawkes_params(events, T=current_min)
        report.params = params
        report.method = "tick" if TICK_OK else ("mle_scipy" if SCIPY_OK else "heuristic")

        # Anlık yoğunluk
        report.current_intensity = round(
            hawkes_intensity(current_min, events, params.mu, params.alpha, params.beta),
            6,
        )
        report.baseline_intensity = round(params.mu, 6)
        report.excitement_ratio = round(
            report.current_intensity / max(params.mu, 1e-6), 4,
        )

        # Son olay
        report.last_event_min = round(float(events[-1]), 1)
        report.time_since_last = round(current_min - events[-1], 1)

        # Gelecek olay olasılığı (1 - survival function)
        for dt, attr in [(5, "next_event_prob_5min"), (10, "next_event_prob_10min")]:
            avg_intensity = (
                report.current_intensity + hawkes_intensity(
                    current_min + dt, events, params.mu, params.alpha, params.beta,
                )
            ) / 2
            prob = 1 - np.exp(-avg_intensity * dt / self._T)
            setattr(report, attr, round(float(np.clip(prob, 0, 1)), 4))

        # Kritiklik
        br = params.branching_ratio
        if br >= self.EXPLOSIVE_THRESHOLD:
            report.criticality = "supercritical"
            report.momentum_level = "explosive"
        elif br >= self.CRITICAL_THRESHOLD:
            report.criticality = "critical"
            report.momentum_level = "high"
        elif br >= 0.5:
            report.criticality = "subcritical"
            report.momentum_level = "medium"
        else:
            report.criticality = "subcritical"
            report.momentum_level = "low"

        # Sinyaller
        if report.momentum_level in ("high", "explosive"):
            report.over_signal = True
        if len(events) >= 2:
            gaps = np.diff(events)
            if len(gaps) > 0 and np.min(gaps) < 5:
                report.goal_burst_alert = True

        report.recommendation = self._advice(report)
        return report

    def predict_next_event(self, event_times: list[float],
                             current_min: float,
                             horizon_min: float = 10.0) -> dict:
        """Bir sonraki olayın ne zaman geleceğini tahmin et."""
        events = np.array(sorted(event_times), dtype=np.float64)
        params = estimate_hawkes_params(events, T=current_min)

        # Thinning algoritması ile simülasyon
        probs = []
        for dt in range(1, int(horizon_min) + 1):
            t = current_min + dt
            lam = hawkes_intensity(t, events, params.mu, params.alpha, params.beta)
            prob = 1 - np.exp(-lam / self._T)
            probs.append(round(float(np.clip(prob, 0, 1)), 4))

        return {
            "minute_probs": probs,
            "peak_minute": int(np.argmax(probs)) + 1,
            "peak_prob": float(max(probs)) if probs else 0.0,
            "branching_ratio": params.branching_ratio,
        }

    def _advice(self, r: HawkesReport) -> str:
        if r.goal_burst_alert:
            return (
                f"GOL PATLAMASI: Ardışık olaylar tespit! "
                f"BR={r.params.branching_ratio:.2f}, "
                f"5dk prob={r.next_event_prob_5min:.0%}. "
                f"ÜST BAHİS SİNYALİ!"
            )
        if r.momentum_level == "explosive":
            return (
                f"PATLAMA NOKTASI: BR={r.params.branching_ratio:.2f} ≈ 1.0! "
                f"Maç kontrolden çıkıyor. Üst bahis güçlü sinyal."
            )
        if r.momentum_level == "high":
            return (
                f"Yüksek momentum: BR={r.params.branching_ratio:.2f}, "
                f"yoğunluk x{r.excitement_ratio:.1f}. "
                f"Üst bahis düşünülebilir."
            )
        return (
            f"Sakin maç: BR={r.params.branching_ratio:.2f}, "
            f"momentum={r.momentum_level}."
        )
