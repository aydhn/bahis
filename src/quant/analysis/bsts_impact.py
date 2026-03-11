"""
bsts_impact.py – Bayesian Structural Time Series (BSTS) ile Müdahale Analizi.

LSTM trendi görür ama "Kırılmayı" (Structural Break) göremez.
BSTS, "Teknik Direktör değiştiği an" veya "Yıldız oyuncu sakatlandığında"
takımın karakterinin nasıl değiştiğini modeller.

Kavramlar:
  - Structural Break: Zaman serisindeki kalıcı kırılma noktası
  - Intervention Analysis: Bir olayın etkisini izole etme
  - Counterfactual: "Olay olmasaydı ne olurdu?" senaryosu
  - Bayesian Posterior: Etkinin olasılık dağılımı
  - Trend + Mevsimsellik + Regresyon ayrıştırma

Sorular:
  "Hoca değişimi kalıcı etki yarattı mı, geçici mi?"
  "Bu sakatlık gol beklentisini ne kadar düşürdü?"
  "Takım kadrosundaki değişim oranları nasıl etkiledi?"

Teknoloji: CausalImpact (R portlu) veya statsmodels + scipy
Fallback: Basit segmented regression + z-test
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from loguru import logger

try:
    from scipy import stats as sp_stats
    from scipy.signal import find_peaks
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.seasonal import seasonal_decompose
    SM_OK = True
except ImportError:
    SM_OK = False

try:
    from causalimpact import CausalImpact as _CausalImpact
    CI_OK = True
except ImportError:
    CI_OK = False


@dataclass
class BreakPoint:
    """Yapısal kırılma noktası."""
    index: int = 0
    date: str = ""
    event: str = ""                 # coach_change | key_injury | transfer | relegation_zone
    pre_mean: float = 0.0           # Kırılma öncesi ortalama
    post_mean: float = 0.0          # Kırılma sonrası ortalama
    change_pct: float = 0.0         # Değişim yüzdesi
    is_significant: bool = False    # İstatistiksel anlamlılık
    p_value: float = 1.0
    confidence_interval: tuple = (0.0, 0.0)


@dataclass
class InterventionEffect:
    """Müdahale etki raporu."""
    intervention: str = ""
    target_metric: str = ""         # goals | xg | points | clean_sheets
    # Causal Impact sonuçları
    avg_effect: float = 0.0         # Ortalama etki büyüklüğü
    cumulative_effect: float = 0.0  # Kümülatif etki
    relative_effect_pct: float = 0.0  # Yüzde etki
    posterior_prob: float = 0.0     # Bayesian posterior olasılık
    # Counterfactual
    predicted_without: float = 0.0  # Müdahale olmasaydı beklenen
    actual: float = 0.0             # Gerçekleşen
    # Karar
    is_significant: bool = False
    is_permanent: bool = False      # Kalıcı mı, geçici mi?
    half_life: float = 0.0          # Geçiciyse: yarı ömrü (maç sayısı)
    method: str = ""
    recommendation: str = ""


@dataclass
class DecompositionResult:
    """Zaman serisi ayrıştırma sonucu."""
    trend: np.ndarray | None = None
    seasonal: np.ndarray | None = None
    residual: np.ndarray | None = None
    break_points: list[BreakPoint] = field(default_factory=list)
    trend_direction: str = "flat"   # rising | falling | flat
    seasonality_strength: float = 0.0


# ═══════════════════════════════════════════════
#  YAPISAL KIRILMA TESPİTİ
# ═══════════════════════════════════════════════
def detect_structural_breaks(data: np.ndarray,
                              min_segment: int = 5,
                              significance: float = 0.05
                              ) -> list[BreakPoint]:
    """CUSUM + t-test ile yapısal kırılma noktalarını bul."""
    if len(data) < min_segment * 2:
        return []

    breaks = []
    n = len(data)

    # Her olası kırılma noktasını test et
    best_stat = 0
    best_idx = -1

    for i in range(min_segment, n - min_segment):
        pre = data[:i]
        post = data[i:]

        if SCIPY_OK:
            t_stat, p_val = sp_stats.ttest_ind(pre, post, equal_var=False)
        else:
            pre_mean = np.mean(pre)
            post_mean = np.mean(post)
            pre_std = np.std(pre, ddof=1)
            post_std = np.std(post, ddof=1)
            se = np.sqrt(pre_std ** 2 / len(pre) + post_std ** 2 / len(post))
            t_stat = (pre_mean - post_mean) / max(se, 1e-10)
            p_val = 0.05  # Basit tahmini

        if abs(t_stat) > abs(best_stat):
            best_stat = t_stat
            best_idx = i
            best_p = p_val if SCIPY_OK else p_val

    if best_idx > 0:
        pre = data[:best_idx]
        post = data[best_idx:]
        pre_mean = float(np.mean(pre))
        post_mean = float(np.mean(post))
        change = (post_mean - pre_mean) / max(abs(pre_mean), 1e-10) * 100

        bp = BreakPoint(
            index=best_idx,
            pre_mean=round(pre_mean, 4),
            post_mean=round(post_mean, 4),
            change_pct=round(change, 2),
            is_significant=best_p < significance,
            p_value=round(best_p, 6),
        )

        # Güven aralığı
        if SCIPY_OK:
            ci = sp_stats.t.interval(
                0.95, len(post) - 1,
                loc=post_mean,
                scale=sp_stats.sem(post),
            )
            bp.confidence_interval = (round(ci[0], 4), round(ci[1], 4))

        breaks.append(bp)

    return breaks


def detect_change_permanence(data: np.ndarray, break_idx: int,
                               window: int = 10) -> tuple[bool, float]:
    """Değişimin kalıcı mı geçici mi olduğunu belirle.

    Returns: (is_permanent, half_life)
    """
    if break_idx >= len(data) - 3:
        return True, 0.0

    post = data[break_idx:]
    if len(post) < 5:
        return True, 0.0

    pre_mean = float(np.mean(data[:break_idx]))
    initial_shock = float(post[0]) - pre_mean

    if abs(initial_shock) < 1e-8:
        return True, 0.0

    # Yarı ömür: etkinin yarıya düştüğü nokta
    half_target = pre_mean + initial_shock * 0.5

    for i, val in enumerate(post):
        if (initial_shock > 0 and val < half_target) or \
           (initial_shock < 0 and val > half_target):
            return False, float(i)

    return True, float(len(post))


# ═══════════════════════════════════════════════
#  BSTS IMPACT ANALYZER
# ═══════════════════════════════════════════════
class BSTSImpactAnalyzer:
    """Bayesian Structural Time Series ile müdahale analizi.

    Kullanım:
        analyzer = BSTSImpactAnalyzer()

        # Performans verisini yükle
        data = [1.2, 1.5, 0.8, 1.1, ...]  # xG serisi

        # Hoca değişimi etkisi (10. maçta)
        effect = analyzer.analyze_intervention(
            data=data,
            intervention_idx=10,
            intervention_name="Hoca Değişimi",
            metric="xG",
        )

        # Yapısal kırılma tespiti
        breaks = analyzer.detect_breaks(data)

        # Zaman serisi ayrıştırma
        decomp = analyzer.decompose(data)
    """

    def __init__(self, significance_level: float = 0.05):
        self._significance = significance_level
        logger.debug("[BSTS] Impact Analyzer başlatıldı.")

    # ═══════════════════════════════════════════
    #  MÜDAHALEAnalizi
    # ═══════════════════════════════════════════
    def analyze_intervention(self, data: list[float] | np.ndarray,
                               intervention_idx: int,
                               intervention_name: str = "",
                               metric: str = "performance",
                               ) -> InterventionEffect:
        """Müdahale etkisini analiz et.

        Args:
            data: Zaman serisi (maçlar bazında)
            intervention_idx: Müdahalenin gerçekleştiği indeks
            intervention_name: "Hoca değişimi", "Sakatlık" vb.
            metric: Ölçülen metrik
        """
        arr = np.array(data, dtype=np.float64)
        effect = InterventionEffect(
            intervention=intervention_name,
            target_metric=metric,
        )

        if intervention_idx <= 2 or intervention_idx >= len(arr) - 2:
            effect.recommendation = "Yetersiz veri – müdahale analiz edilemedi."
            effect.method = "insufficient"
            return effect

        pre = arr[:intervention_idx]
        post = arr[intervention_idx:]

        # ── CausalImpact (en iyi yöntem) ──
        if CI_OK and len(arr) > 10:
            effect = self._causal_impact_analysis(
                arr, intervention_idx, effect,
            )
        elif SM_OK and len(arr) > 10:
            # ── Holt-Winters Counterfactual ──
            effect = self._holt_winters_counterfactual(
                arr, intervention_idx, effect,
            )
        else:
            # ── Basit t-test ──
            effect = self._simple_ttest(pre, post, effect)

        # ── Kalıcılık kontrolü ──
        is_permanent, half_life = detect_change_permanence(
            arr, intervention_idx,
        )
        effect.is_permanent = is_permanent
        effect.half_life = round(half_life, 1)

        # ── Tavsiye ──
        effect.recommendation = self._generate_advice(effect)

        return effect

    def _causal_impact_analysis(self, data: np.ndarray,
                                  break_idx: int,
                                  effect: InterventionEffect
                                  ) -> InterventionEffect:
        """CausalImpact kütüphanesi ile Bayesian analiz."""
        try:
            import pandas as pd
            df = pd.DataFrame({"y": data})
            pre_period = [0, break_idx - 1]
            post_period = [break_idx, len(data) - 1]

            ci = _CausalImpact(df, pre_period, post_period)
            summary = ci.summary_data

            if summary is not None:
                effect.avg_effect = round(float(
                    summary.get("average", {}).get("abs_effect", 0)
                ), 4)
                effect.cumulative_effect = round(float(
                    summary.get("cumulative", {}).get("abs_effect", 0)
                ), 4)
                effect.relative_effect_pct = round(float(
                    summary.get("average", {}).get("rel_effect", 0)
                ) * 100, 2)
                effect.posterior_prob = round(float(
                    summary.get("average", {}).get("p_value", 1)
                ), 4)
                effect.is_significant = effect.posterior_prob < self._significance

            effect.method = "causal_impact"
        except Exception as e:
            logger.debug(f"[BSTS] CausalImpact hatası: {e}")
            pre = data[:break_idx]
            post = data[break_idx:]
            effect = self._simple_ttest(pre, post, effect)

        return effect

    def _holt_winters_counterfactual(self, data: np.ndarray,
                                       break_idx: int,
                                       effect: InterventionEffect
                                       ) -> InterventionEffect:
        """Holt-Winters ile counterfactual tahmin."""
        try:
            pre = data[:break_idx]
            post = data[break_idx:]

            # Pre-period modeli
            model = ExponentialSmoothing(
                pre, trend="add", seasonal=None,
                initialization_method="estimated",
            ).fit(optimized=True)

            # Müdahale olmasaydı tahmin
            forecast = model.forecast(len(post))
            effect.predicted_without = round(float(np.mean(forecast)), 4)
            effect.actual = round(float(np.mean(post)), 4)

            # Etki
            effect.avg_effect = round(
                effect.actual - effect.predicted_without, 4,
            )
            if effect.predicted_without != 0:
                effect.relative_effect_pct = round(
                    (effect.avg_effect / abs(effect.predicted_without)) * 100, 2,
                )
            effect.cumulative_effect = round(
                float(np.sum(post - forecast[:len(post)])), 4,
            )

            # Anlamlılık: tahmin hatası vs etki
            residuals = pre - model.fittedvalues
            std_err = float(np.std(residuals))
            if std_err > 0:
                z_score = effect.avg_effect / std_err
                if SCIPY_OK:
                    effect.posterior_prob = round(
                        2 * (1 - sp_stats.norm.cdf(abs(z_score))), 4,
                    )
                else:
                    effect.posterior_prob = 0.05 if abs(z_score) > 1.96 else 0.5
                effect.is_significant = effect.posterior_prob < self._significance

            effect.method = "holt_winters"
        except Exception as e:
            logger.debug(f"[BSTS] Holt-Winters hatası: {e}")
            pre = data[:break_idx]
            post = data[break_idx:]
            effect = self._simple_ttest(pre, post, effect)

        return effect

    def _simple_ttest(self, pre: np.ndarray, post: np.ndarray,
                       effect: InterventionEffect) -> InterventionEffect:
        """Basit t-test ile etki analizi."""
        pre_mean = float(np.mean(pre))
        post_mean = float(np.mean(post))

        effect.predicted_without = round(pre_mean, 4)
        effect.actual = round(post_mean, 4)
        effect.avg_effect = round(post_mean - pre_mean, 4)

        if pre_mean != 0:
            effect.relative_effect_pct = round(
                (post_mean - pre_mean) / abs(pre_mean) * 100, 2,
            )

        if SCIPY_OK:
            t_stat, p_val = sp_stats.ttest_ind(pre, post, equal_var=False)
            effect.posterior_prob = round(float(p_val), 4)
            effect.is_significant = p_val < self._significance
        else:
            effect.posterior_prob = 0.5

        effect.method = "t_test"
        return effect

    # ═══════════════════════════════════════════
    #  KIRILMA TESPİTİ
    # ═══════════════════════════════════════════
    def detect_breaks(self, data: list[float] | np.ndarray,
                       min_segment: int = 5) -> list[BreakPoint]:
        """Yapısal kırılma noktalarını tespit et."""
        arr = np.array(data, dtype=np.float64)
        return detect_structural_breaks(arr, min_segment, self._significance)

    # ═══════════════════════════════════════════
    #  ZAMAN SERİSİ AYRIŞTIRMA
    # ═══════════════════════════════════════════
    def decompose(self, data: list[float] | np.ndarray,
                    period: int = 5) -> DecompositionResult:
        """Trend + Mevsimsellik + Gürültü ayrıştırma."""
        result = DecompositionResult()
        arr = np.array(data, dtype=np.float64)

        if SM_OK and len(arr) >= period * 2:
            try:
                import pandas as pd
                series = pd.Series(arr)
                decomp = seasonal_decompose(
                    series, model="additive", period=period,
                )
                result.trend = decomp.trend.values
                result.seasonal = decomp.seasonal.values
                result.residual = decomp.resid.values

                # Trend yönü
                valid_trend = result.trend[~np.isnan(result.trend)]
                if len(valid_trend) > 3:
                    slope = np.polyfit(range(len(valid_trend)), valid_trend, 1)[0]
                    if slope > 0.01:
                        result.trend_direction = "rising"
                    elif slope < -0.01:
                        result.trend_direction = "falling"

                # Mevsimsellik gücü
                if result.seasonal is not None:
                    valid_s = result.seasonal[~np.isnan(result.seasonal)]
                    if len(valid_s) > 0:
                        result.seasonality_strength = round(
                            float(np.std(valid_s) / max(np.std(arr), 1e-8)), 4,
                        )
            except Exception as e:
                logger.debug(f"Exception caught: {e}")

        # Kırılma tespiti
        result.break_points = self.detect_breaks(arr)

        return result

    # ═══════════════════════════════════════════
    #  TAVSİYE
    # ═══════════════════════════════════════════
    def _generate_advice(self, effect: InterventionEffect) -> str:
        """Müdahale etkisi tavsiyesi."""
        parts = []

        if not effect.is_significant:
            parts.append(
                f"İstatistiksel olarak anlamlı DEĞİL (p={effect.posterior_prob:.3f}). "
                f"Gürültü olabilir."
            )
        else:
            direction = "ARTIŞ" if effect.avg_effect > 0 else "DÜŞÜŞ"
            parts.append(
                f"{direction}: {effect.relative_effect_pct:+.1f}% "
                f"(p={effect.posterior_prob:.3f})"
            )

        if effect.is_permanent:
            parts.append("Etki KALICI görünüyor.")
        else:
            parts.append(
                f"Etki GEÇİCİ (yarı ömür: ~{effect.half_life:.0f} maç)."
            )

        return " | ".join(parts)
