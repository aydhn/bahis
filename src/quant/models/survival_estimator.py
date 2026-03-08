"""
survival_estimator.py – Survival Analysis (Sağkalım Analizi).

Poisson "Gol olur mu?" der. Sağkalım Analizi ise
"Savunma hattı daha ne kadar dayanabilir?" sorusunu cevaplar.

Kavramlar:
  - Time-to-Concede: Gol yeme süresi (dakika cinsinden)
  - Survival Function S(t): t dakikasında hâlâ gol yememiş olma olasılığı
  - Hazard Function h(t): t anındaki anlık gol yeme riski
  - Cumulative Hazard H(t): 0'dan t'ye toplam birikmiş tehlike
  - Nelson-Aalen Estimator: H(t) için non-parametrik tahmin
  - Kaplan-Meier: S(t) için non-parametrik tahmin
  - Cox Proportional Hazards: Kovaryatlarla tehlike modellemesi

Sinyaller:
  - H(t) > 1.5 → "Baraj yıkılıyor!" – gol an meselesi
  - S(t) < 0.3 → "Kale düşmek üzere" – canlı bahis sinyali
  - Hazard ratio > 2.0 → Baskı altındaki takım tehlikede

Teknoloji: lifelines (Python Survival Analysis)
Fallback: Manuel Kaplan-Meier + Nelson-Aalen (numpy)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from loguru import logger

try:
    from lifelines import (
        KaplanMeierFitter,
        NelsonAalenFitter,
        CoxPHFitter,
    )
    LIFELINES_OK = True
except ImportError:
    LIFELINES_OK = False
    logger.debug("lifelines yüklü değil – manuel sağkalım fallback.")

try:
    import pandas as pd
    PANDAS_OK = True
except ImportError:
    PANDAS_OK = False


# ═══════════════════════════════════════════════
#  VERİ YAPILARI
# ═══════════════════════════════════════════════
@dataclass
class SurvivalParams:
    """Sağkalım modeli parametreleri."""
    median_survival: float = 0.0     # Medyan dayanma süresi (dk)
    mean_survival: float = 0.0       # Ortalama dayanma süresi
    current_hazard: float = 0.0      # Anlık tehlike oranı
    cumulative_hazard: float = 0.0   # Birikmiş tehlike
    survival_prob: float = 1.0       # Sağkalım olasılığı S(t)
    hazard_slope: float = 0.0        # Tehlike eğimi (dikleşme)


@dataclass
class SurvivalReport:
    """Sağkalım analizi raporu."""
    team: str = ""
    match_id: str = ""
    # Model çıktıları
    params: SurvivalParams = field(default_factory=SurvivalParams)
    # Mevcut durum
    current_minute: float = 0.0
    minutes_since_last_goal: float = 0.0
    # Tahmin
    prob_concede_5min: float = 0.0    # 5dk içinde gol yeme olasılığı
    prob_concede_10min: float = 0.0
    prob_concede_15min: float = 0.0
    expected_time_to_goal: float = 0.0  # Beklenen gol süresi
    # Sinyaller
    dam_breaking: bool = False        # "Baraj Yıkılıyor" sinyali
    fortress_mode: bool = False       # Takım çok sağlam (S(t) > 0.8)
    hazard_spike: bool = False        # Tehlike ani artış
    # Rejim
    risk_level: str = "low"           # "low" | "medium" | "high" | "critical"
    over_signal: bool = False         # Üst bahis sinyali
    under_signal: bool = False        # Alt bahis sinyali
    # Meta
    method: str = ""
    recommendation: str = ""


# ═══════════════════════════════════════════════
#  MANUEL KAPLAN-MEIER & NELSON-AALEN
# ═══════════════════════════════════════════════
def manual_kaplan_meier(durations: np.ndarray,
                          observed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Manuel Kaplan-Meier sağkalım tahmini.

    S(t) = Π (1 - dᵢ/nᵢ) [tᵢ ≤ t]
    """
    order = np.argsort(durations)
    t_sorted = durations[order]
    o_sorted = observed[order]

    unique_times = np.unique(t_sorted)
    S = np.ones(len(unique_times) + 1)
    times = np.zeros(len(unique_times) + 1)
    times[0] = 0.0
    n_at_risk = len(durations)

    for i, t in enumerate(unique_times):
        mask = t_sorted == t
        d_i = int(o_sorted[mask].sum())       # olay sayısı
        n_i = int(mask.sum())                  # toplam
        if n_at_risk > 0:
            S[i + 1] = S[i] * (1 - d_i / n_at_risk)
        else:
            S[i + 1] = S[i]
        times[i + 1] = t
        n_at_risk -= n_i

    return times, np.clip(S, 0, 1)


def manual_nelson_aalen(durations: np.ndarray,
                           observed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Manuel Nelson-Aalen kümülatif tehlike tahmini.

    H(t) = Σ (dᵢ/nᵢ) [tᵢ ≤ t]
    """
    order = np.argsort(durations)
    t_sorted = durations[order]
    o_sorted = observed[order]

    unique_times = np.unique(t_sorted)
    H = np.zeros(len(unique_times) + 1)
    times = np.zeros(len(unique_times) + 1)
    times[0] = 0.0
    n_at_risk = len(durations)

    for i, t in enumerate(unique_times):
        mask = t_sorted == t
        d_i = int(o_sorted[mask].sum())
        n_i = int(mask.sum())
        if n_at_risk > 0:
            H[i + 1] = H[i] + d_i / n_at_risk
        else:
            H[i + 1] = H[i]
        times[i + 1] = t
        n_at_risk -= n_i

    return times, H


# ═══════════════════════════════════════════════
#  SURVIVAL ESTIMATOR (Ana Sınıf)
# ═══════════════════════════════════════════════
class SurvivalEstimator:
    """Sağkalım analizi ile gol yeme süresini tahmin eder.

    Kullanım:
        se = SurvivalEstimator()

        # Geçmiş maç verileri: gol yemeden geçen süreler
        # duration: dakika, observed: gol yendi mi? (1=evet, 0=maç bitti/gol yemedi)
        durations = [23, 45, 67, 12, 90, 34, 55, 78, 90, 28]
        observed  = [ 1,  1,  1,  1,  0,  1,  1,  1,  0,  1]

        se.fit(durations, observed)

        # Mevcut maç: 65. dakikada, son golü 50. dakikada yedi
        report = se.analyze(
            current_minute=65,
            last_goal_minute=50,
            team="Galatasaray",
            match_id="gs_fb",
        )

        if report.dam_breaking:
            place_over_bet()
    """

    DAM_THRESHOLD = 1.5        # Kümülatif tehlike eşiği
    FORTRESS_THRESHOLD = 0.8   # Sağkalım > bu → kale sağlam
    HAZARD_SPIKE_MULT = 2.0    # Tehlike ani artış çarpanı

    def __init__(self):
        self._km_fitter = None
        self._na_fitter = None
        self._cox_fitter = None
        self._fitted = False
        self._durations: np.ndarray | None = None
        self._observed: np.ndarray | None = None
        # Manuel fallback verileri
        self._km_times: np.ndarray | None = None
        self._km_survival: np.ndarray | None = None
        self._na_times: np.ndarray | None = None
        self._na_hazard: np.ndarray | None = None

        logger.debug("[Survival] Estimator başlatıldı.")

    def fit(self, durations: list[float] | np.ndarray,
            observed: list[int] | np.ndarray | None = None,
            covariates: dict[str, list] | None = None) -> None:
        """Geçmiş verilerle modeli eğit.

        Args:
            durations: Gol yemeye kadar geçen süreler (dakika)
            observed: Gol yendi mi? (1=evet, 0=sansürlü/maç bitti)
            covariates: Ek değişkenler (Cox modeli için)
        """
        self._durations = np.array(durations, dtype=np.float64)
        if observed is None:
            self._observed = np.ones_like(self._durations, dtype=np.int32)
        else:
            self._observed = np.array(observed, dtype=np.int32)

        if LIFELINES_OK:
            try:
                # Kaplan-Meier
                self._km_fitter = KaplanMeierFitter()
                self._km_fitter.fit(self._durations, self._observed)

                # Nelson-Aalen
                self._na_fitter = NelsonAalenFitter()
                self._na_fitter.fit(self._durations, self._observed)

                # Cox PH (kovaryatlar varsa)
                if covariates and PANDAS_OK:
                    df = pd.DataFrame(covariates)
                    df["duration"] = self._durations
                    df["observed"] = self._observed
                    self._cox_fitter = CoxPHFitter()
                    self._cox_fitter.fit(df, "duration", "observed")

                self._fitted = True
                logger.debug(
                    f"[Survival] lifelines fit tamamlandı: "
                    f"n={len(self._durations)}, "
                    f"median={self._km_fitter.median_survival_time_:.1f}dk"
                )
                return
            except Exception as e:
                logger.debug(f"[Survival] lifelines hatası: {e}")

        # Manuel fallback
        self._km_times, self._km_survival = manual_kaplan_meier(
            self._durations, self._observed,
        )
        self._na_times, self._na_hazard = manual_nelson_aalen(
            self._durations, self._observed,
        )
        self._fitted = True
        logger.debug(
            f"[Survival] Manuel fit tamamlandı: n={len(self._durations)}"
        )

    def analyze(self, current_minute: float = 0.0,
                last_goal_minute: float | None = None,
                team: str = "",
                match_id: str = "",
                pressure_index: float = 0.0) -> SurvivalReport:
        """Mevcut maç durumunu analiz et."""
        report = SurvivalReport(team=team, match_id=match_id)
        report.current_minute = current_minute

        if not self._fitted:
            report.recommendation = "Model henüz eğitilmedi. fit() çağırın."
            report.method = "not_fitted"
            return report

        # Son golden bu yana geçen süre
        if last_goal_minute is not None:
            report.minutes_since_last_goal = current_minute - last_goal_minute
        else:
            report.minutes_since_last_goal = current_minute

        t = report.minutes_since_last_goal

        # Sağkalım ve tehlike hesapla
        params = SurvivalParams()

        if LIFELINES_OK and self._km_fitter and self._na_fitter:
            params = self._compute_lifelines(t, current_minute, pressure_index)
            report.method = "lifelines"
        else:
            params = self._compute_manual(t)
            report.method = "manual_km_na"

        report.params = params

        # Gelecek olasılıkları
        for dt, attr in [(5, "prob_concede_5min"),
                          (10, "prob_concede_10min"),
                          (15, "prob_concede_15min")]:
            s_now = params.survival_prob
            s_future = self._survival_at(t + dt)
            # P(gol [t, t+dt]) = 1 - S(t+dt)/S(t)
            if s_now > 0:
                prob = 1 - s_future / s_now
            else:
                prob = 1.0
            setattr(report, attr, round(float(np.clip(prob, 0, 1)), 4))

        # Beklenen gol süresi
        report.expected_time_to_goal = round(
            self._expected_remaining(t), 1,
        )

        # Sinyaller
        if params.cumulative_hazard > self.DAM_THRESHOLD:
            report.dam_breaking = True
        if params.survival_prob > self.FORTRESS_THRESHOLD:
            report.fortress_mode = True
        if params.hazard_slope > self.HAZARD_SPIKE_MULT:
            report.hazard_spike = True

        # Risk seviyesi
        if report.dam_breaking or report.prob_concede_5min > 0.5:
            report.risk_level = "critical"
        elif report.prob_concede_10min > 0.5 or report.hazard_spike:
            report.risk_level = "high"
        elif report.prob_concede_10min > 0.3:
            report.risk_level = "medium"
        else:
            report.risk_level = "low"

        # Bahis sinyalleri
        if report.risk_level in ("critical", "high"):
            report.over_signal = True
        elif report.fortress_mode and report.risk_level == "low":
            report.under_signal = True

        report.recommendation = self._advice(report)
        return report

    def _compute_lifelines(self, t: float, current_min: float,
                             pressure: float) -> SurvivalParams:
        """lifelines ile parametreleri hesapla."""
        params = SurvivalParams()

        try:
            params.median_survival = float(
                self._km_fitter.median_survival_time_
            )
        except Exception:
            params.median_survival = 45.0

        # S(t) – sağkalım olasılığı
        try:
            sf = self._km_fitter.predict(t)
            params.survival_prob = round(float(sf.iloc[0]) if hasattr(sf, 'iloc') else float(sf), 4)
        except Exception:
            params.survival_prob = max(0, 1 - t / 90)

        # H(t) – kümülatif tehlike
        try:
            ch = self._na_fitter.predict(t)
            params.cumulative_hazard = round(float(ch.iloc[0]) if hasattr(ch, 'iloc') else float(ch), 4)
        except Exception:
            params.cumulative_hazard = -np.log(max(params.survival_prob, 1e-6))

        # h(t) – anlık tehlike (H(t) türevi yaklaşımı)
        try:
            ch_prev = self._na_fitter.predict(max(0, t - 5))
            ch_curr = self._na_fitter.predict(t)
            h_prev = float(ch_prev.iloc[0]) if hasattr(ch_prev, 'iloc') else float(ch_prev)
            h_curr = float(ch_curr.iloc[0]) if hasattr(ch_curr, 'iloc') else float(ch_curr)
            params.current_hazard = round(max(0, (h_curr - h_prev) / 5), 6)
        except Exception:
            params.current_hazard = 0.0

        # Tehlike eğimi (son 10 dk ile önceki 10 dk karşılaştırması)
        try:
            ch_10_ago = self._na_fitter.predict(max(0, t - 10))
            ch_20_ago = self._na_fitter.predict(max(0, t - 20))
            h10 = float(ch_10_ago.iloc[0]) if hasattr(ch_10_ago, 'iloc') else float(ch_10_ago)
            h20 = float(ch_20_ago.iloc[0]) if hasattr(ch_20_ago, 'iloc') else float(ch_20_ago)
            h_curr = params.cumulative_hazard
            slope_recent = h_curr - h10
            slope_prev = h10 - h20
            if slope_prev > 0:
                params.hazard_slope = round(slope_recent / slope_prev, 2)
        except Exception:
            params.hazard_slope = 1.0

        # Baskı indexi (pressure) artıyorsa tehlikeyi çarpanla
        if pressure > 0:
            params.current_hazard *= (1 + pressure * 0.5)

        return params

    def _compute_manual(self, t: float) -> SurvivalParams:
        """Manuel hesaplama (numpy)."""
        params = SurvivalParams()

        if self._km_times is not None and self._km_survival is not None:
            idx = np.searchsorted(self._km_times, t, side="right") - 1
            idx = np.clip(idx, 0, len(self._km_survival) - 1)
            params.survival_prob = round(float(self._km_survival[idx]), 4)

        if self._na_times is not None and self._na_hazard is not None:
            idx = np.searchsorted(self._na_times, t, side="right") - 1
            idx = np.clip(idx, 0, len(self._na_hazard) - 1)
            params.cumulative_hazard = round(float(self._na_hazard[idx]), 4)

            # Anlık tehlike (fark)
            if idx > 0:
                dt = self._na_times[idx] - self._na_times[idx - 1]
                if dt > 0:
                    params.current_hazard = round(
                        (self._na_hazard[idx] - self._na_hazard[idx - 1]) / dt,
                        6,
                    )

        # Medyan
        if self._km_survival is not None:
            below_50 = np.where(self._km_survival <= 0.5)[0]
            if len(below_50) > 0:
                params.median_survival = float(self._km_times[below_50[0]])
            else:
                params.median_survival = 90.0

        params.hazard_slope = 1.0
        return params

    def _survival_at(self, t: float) -> float:
        """S(t) değerini döndür."""
        if LIFELINES_OK and self._km_fitter:
            try:
                sf = self._km_fitter.predict(max(0, t))
                return float(sf.iloc[0]) if hasattr(sf, 'iloc') else float(sf)
            except Exception:
                pass

        if self._km_times is not None and self._km_survival is not None:
            idx = np.searchsorted(self._km_times, t, side="right") - 1
            idx = np.clip(idx, 0, len(self._km_survival) - 1)
            return float(self._km_survival[idx])

        return max(0, 1 - t / 90)

    def _expected_remaining(self, t_current: float) -> float:
        """Beklenen kalan dayanma süresi.

        E[T - t | T > t] ≈ ∫ S(u)/S(t) du [t, 90]
        """
        s_now = self._survival_at(t_current)
        if s_now <= 1e-6:
            return 0.0

        # Trapez yaklaşımı
        grid = np.linspace(t_current, 90, 50)
        s_values = np.array([self._survival_at(u) for u in grid])
        integrand = s_values / s_now
        expected = float(np.trapz(integrand, grid))
        return max(0, expected)

    def _advice(self, r: SurvivalReport) -> str:
        if r.dam_breaking:
            return (
                f"BARAJ YIKILIYOR: H(t)={r.params.cumulative_hazard:.2f} > "
                f"{self.DAM_THRESHOLD}, S(t)={r.params.survival_prob:.0%}. "
                f"5dk prob={r.prob_concede_5min:.0%}. ÜST BAHİS SİNYALİ!"
            )
        if r.hazard_spike:
            return (
                f"TEHLİKE SIVRILMASI: Eğim x{r.params.hazard_slope:.1f}. "
                f"Savunma hattı bozuluyor. Üst bahis düşünülebilir."
            )
        if r.fortress_mode:
            return (
                f"KALE SAĞLAM: S(t)={r.params.survival_prob:.0%}, "
                f"düşük tehlike. Alt bahis sinyali."
            )
        return (
            f"Normal risk: S(t)={r.params.survival_prob:.0%}, "
            f"beklenen gol={r.expected_time_to_goal:.0f}dk."
        )
