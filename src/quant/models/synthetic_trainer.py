"""
synthetic_trainer.py – Synthetic Data Vault (Sentetik Veri Üretimi).

Haftada 10 maç = yetersiz veri. Eldeki verinin "DNA'sını"
kopyalayarak istatistiksel olarak ayırt edilemez
sentetik maçlar üretir. Botunuz 10 yıllık tecrübeyi
1 gecede kazanır.

Kavramlar:
  - Synthetic Data: Gerçek veriden üretilen sahte ama istatistiksel
    olarak eşdeğer veri
  - GaussianCopula: Değişkenler arası korelasyonu koruyarak
    marjinal dağılımları modeller
  - CTGAN: Conditional Tabular GAN – derin öğrenme tabanlı
  - Privacy Preserving: Gerçek veri paylaşılmaz, sadece dağılım
  - Data Augmentation: Eğitim verisini büyütme
  - Kolmogorov-Smirnov Test: Gerçek vs sentetik karşılaştırma

Akış:
  1. Gerçek maç verisini yükle
  2. SDV modeli fit et (dağılımları öğren)
  3. N tane sentetik maç üret
  4. Kalite kontrolü (KS test, korelasyon karşılaştırma)
  5. RL, Ensemble, LightGBM modellerini sentetik + gerçek ile eğit

Teknoloji: SDV (Synthetic Data Vault)
Fallback: Bootstrap resampling + Gaussian noise injection
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

try:
    from sdv.single_table import GaussianCopulaSynthesizer
    from sdv.metadata import SingleTableMetadata
    SDV_OK = True
except ImportError:
    SDV_OK = False
    logger.debug("sdv yüklü değil – bootstrap fallback.")

try:
    import pandas as pd
    PANDAS_OK = True
except ImportError:
    PANDAS_OK = False

try:
    from scipy.stats import ks_2samp
    KS_OK = True
except ImportError:
    KS_OK = False

ROOT = Path(__file__).resolve().parent.parent.parent
SYNTH_DIR = ROOT / "data" / "synthetic"
SYNTH_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════
#  VERİ YAPILARI
# ═══════════════════════════════════════════════
@dataclass
class QualityMetric:
    """Sentetik veri kalite metriği."""
    column: str = ""
    ks_statistic: float = 0.0    # Kolmogorov-Smirnov test istatistiği
    ks_pvalue: float = 0.0       # p-değeri (> 0.05 → iyi)
    mean_diff_pct: float = 0.0   # Ortalama fark yüzdesi
    std_diff_pct: float = 0.0    # Std sapma fark yüzdesi
    passed: bool = True          # Kalite geçti mi


@dataclass
class SyntheticReport:
    """Sentetik veri üretim raporu."""
    n_real: int = 0              # Gerçek veri satır sayısı
    n_synthetic: int = 0         # Üretilen sentetik satır sayısı
    n_columns: int = 0           # Sütun sayısı
    # Kalite
    quality_metrics: list[QualityMetric] = field(default_factory=list)
    avg_ks_pvalue: float = 0.0   # Ortalama KS p-değeri
    correlation_diff: float = 0.0  # Korelasyon matris farkı
    overall_quality: str = ""    # "excellent" | "good" | "poor"
    # Meta
    generation_time_sec: float = 0.0
    method: str = ""
    output_path: str = ""
    recommendation: str = ""


# ═══════════════════════════════════════════════
#  BOOTSTRAP FALLBACK
# ═══════════════════════════════════════════════
def bootstrap_augment(data: np.ndarray, n_samples: int,
                        noise_scale: float = 0.05) -> np.ndarray:
    """Bootstrap resampling + Gaussian gürültü.

    Gerçek veriden rassal seçim + küçük gürültü ekle.
    """
    n_real = len(data)
    indices = np.random.randint(0, n_real, n_samples)
    resampled = data[indices].copy()

    # Gaussian gürültü (her sütunun std'sine orantılı)
    stds = np.std(data, axis=0)
    noise = np.random.randn(*resampled.shape) * stds * noise_scale
    resampled += noise

    return resampled


# ═══════════════════════════════════════════════
#  SYNTHETIC TRAINER (Ana Sınıf)
# ═══════════════════════════════════════════════
class SyntheticTrainer:
    """Sentetik veri üretimi ve model eğitimi.

    Kullanım:
        st = SyntheticTrainer()

        # Gerçek veri (DataFrame veya ndarray)
        real_data = load_match_data()  # 500 satır

        # 50.000 sentetik maç üret
        synthetic = st.generate(real_data, n_samples=50000)

        # Kalite raporu
        report = st.quality_report(real_data, synthetic)

        # Karma veriyle model eğit
        combined = st.combine(real_data, synthetic, real_weight=2.0)
    """

    def __init__(self, noise_scale: float = 0.03):
        self._noise_scale = noise_scale
        self._synthesizer: Any = None
        self._fitted = False

        logger.debug("[Synthetic] Trainer başlatıldı.")

    def fit(self, data: np.ndarray | Any,
            column_names: list[str] | None = None) -> None:
        """Gerçek veri dağılımını öğren."""
        if SDV_OK and PANDAS_OK:
            try:
                if isinstance(data, np.ndarray):
                    cols = column_names or [f"f{i}" for i in range(data.shape[1])]
                    df = pd.DataFrame(data, columns=cols)
                else:
                    df = data

                metadata = SingleTableMetadata()
                metadata.detect_from_dataframe(df)

                self._synthesizer = GaussianCopulaSynthesizer(metadata)
                self._synthesizer.fit(df)
                self._fitted = True
                logger.debug(
                    f"[Synthetic] SDV fit tamamlandı: "
                    f"{len(df)} satır, {len(df.columns)} sütun"
                )
                return
            except Exception as e:
                logger.debug(f"[Synthetic] SDV hatası: {e}")

        # Fallback: ham veriyi sakla
        if isinstance(data, np.ndarray):
            self._raw_data = data
        elif PANDAS_OK and hasattr(data, "values"):
            self._raw_data = data.values
        else:
            self._raw_data = np.array(data, dtype=np.float64)

        self._fitted = True
        logger.debug("[Synthetic] Bootstrap fallback hazır.")

    def generate(self, data: np.ndarray | Any | None = None,
                   n_samples: int = 50000,
                   column_names: list[str] | None = None) -> np.ndarray:
        """Sentetik veri üret."""
        if not self._fitted and data is not None:
            self.fit(data, column_names)

        if SDV_OK and self._synthesizer:
            try:
                synthetic_df = self._synthesizer.sample(n_samples)
                result = synthetic_df.values.astype(np.float64)
                logger.info(
                    f"[Synthetic] SDV: {n_samples} sentetik satır üretildi."
                )
                return result
            except Exception as e:
                logger.debug(f"[Synthetic] SDV üretim hatası: {e}")

        # Fallback: bootstrap
        raw = getattr(self, "_raw_data", None)
        if raw is None and data is not None:
            if isinstance(data, np.ndarray):
                raw = data
            elif PANDAS_OK and hasattr(data, "values"):
                raw = data.values
            else:
                raw = np.array(data, dtype=np.float64)

        if raw is not None:
            result = bootstrap_augment(raw, n_samples, self._noise_scale)
            logger.info(
                f"[Synthetic] Bootstrap: {n_samples} sentetik satır üretildi."
            )
            return result

        return np.random.randn(n_samples, 10)

    def quality_check(self, real: np.ndarray,
                        synthetic: np.ndarray,
                        column_names: list[str] | None = None) -> SyntheticReport:
        """Sentetik veri kalitesini değerlendir."""
        report = SyntheticReport(
            n_real=len(real),
            n_synthetic=len(synthetic),
            n_columns=real.shape[1] if real.ndim > 1 else 1,
        )

        if real.ndim == 1:
            real = real.reshape(-1, 1)
            synthetic = synthetic.reshape(-1, 1)

        cols = column_names or [f"f{i}" for i in range(real.shape[1])]
        report.method = "sdv" if (SDV_OK and self._synthesizer) else "bootstrap"

        ks_pvalues = []

        for i, col in enumerate(cols):
            if i >= real.shape[1] or i >= synthetic.shape[1]:
                break

            metric = QualityMetric(column=col)
            real_col = real[:, i]
            synth_col = synthetic[:, i]

            # KS test
            if KS_OK:
                try:
                    stat, pval = ks_2samp(real_col, synth_col)
                    metric.ks_statistic = round(float(stat), 4)
                    metric.ks_pvalue = round(float(pval), 4)
                    metric.passed = pval > 0.05
                    ks_pvalues.append(pval)
                except Exception:
                    pass

            # Ortalama/std fark
            r_mean, s_mean = np.mean(real_col), np.mean(synth_col)
            r_std, s_std = np.std(real_col), np.std(synth_col)
            if abs(r_mean) > 1e-6:
                metric.mean_diff_pct = round(
                    abs(s_mean - r_mean) / abs(r_mean) * 100, 2,
                )
            if abs(r_std) > 1e-6:
                metric.std_diff_pct = round(
                    abs(s_std - r_std) / abs(r_std) * 100, 2,
                )

            report.quality_metrics.append(metric)

        # Korelasyon matrisi karşılaştırma
        if real.shape[1] > 1:
            try:
                corr_real = np.corrcoef(real.T)
                corr_synth = np.corrcoef(synthetic[:, :real.shape[1]].T)
                report.correlation_diff = round(
                    float(np.mean(np.abs(corr_real - corr_synth))), 4,
                )
            except Exception:
                pass

        # Genel kalite
        if ks_pvalues:
            report.avg_ks_pvalue = round(float(np.mean(ks_pvalues)), 4)
            pass_rate = sum(1 for p in ks_pvalues if p > 0.05) / len(ks_pvalues)
            if pass_rate > 0.8 and report.correlation_diff < 0.1:
                report.overall_quality = "excellent"
            elif pass_rate > 0.5:
                report.overall_quality = "good"
            else:
                report.overall_quality = "poor"
        else:
            report.overall_quality = "unknown"

        report.recommendation = self._advice(report)
        return report

    def combine(self, real: np.ndarray, synthetic: np.ndarray,
                  real_weight: float = 2.0) -> np.ndarray:
        """Gerçek ve sentetik veriyi birleştir.

        real_weight > 1 → gerçek veri daha fazla tekrarlanır.
        """
        n_repeat = max(1, int(real_weight))
        real_repeated = np.tile(real, (n_repeat, 1)) if real.ndim > 1 else np.tile(real, n_repeat)
        combined = np.vstack([real_repeated, synthetic]) if real.ndim > 1 else np.concatenate([real_repeated, synthetic])
        np.random.shuffle(combined)
        return combined

    def save(self, synthetic: np.ndarray, name: str = "latest") -> Path:
        """Sentetik veriyi kaydet."""
        path = SYNTH_DIR / f"synthetic_{name}.npy"
        np.save(path, synthetic)
        logger.info(f"[Synthetic] Kaydedildi: {path}")
        return path

    def _advice(self, r: SyntheticReport) -> str:
        if r.overall_quality == "excellent":
            return (
                f"Mükemmel kalite: KS p={r.avg_ks_pvalue:.3f}, "
                f"kor. fark={r.correlation_diff:.3f}. "
                f"{r.n_synthetic} sentetik satır kullanılabilir."
            )
        if r.overall_quality == "good":
            return (
                f"İyi kalite: KS p={r.avg_ks_pvalue:.3f}. "
                f"Modelleri eğitmek için uygun."
            )
        return (
            f"Düşük kalite: KS p={r.avg_ks_pvalue:.3f}. "
            f"Gerçek veriyi artırın veya noise_scale'i düşürün."
        )
