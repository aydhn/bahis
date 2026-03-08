"""
fisher_geometry.py – Information Geometry (Fisher Bilgi Metriği).

Olasılık dağılımlarını bir "yüzey" (manifold) üzerinde noktalar
olarak görür. İki dağılım arasındaki "mesafe" KL-divergence değil,
geometrik geodezik mesafedir. Model drift'i, rejim değişimini ve
oran manipülasyonunu çok daha hassas tespit eder.

Kavramlar:
  - Fisher Information Matrix (FIM): Dağılımın "eğriliğini" ölçer
    I_ij(θ) = E[∂logf/∂θ_i · ∂logf/∂θ_j]
  - Fisher-Rao Distance: Manifold üzerindeki geodezik mesafe
  - Riemannian Geometry: Dağılımlar bir Riemann manifoldu oluşturur
  - Natural Gradient: Fisher metriği ile ölçeklenmiş gradyan —
    parametre uzayında daha hızlı ve kararlı optimizasyon
  - KL-Divergence: Asimetrik, Fisher-Rao simetrik (gerçek mesafe)
  - Jeffreys Divergence: Simetrikleştirilmiş KL
  - Anomali Tespiti: Dağılım uzayında "uzak" noktalar = anomali
  - Model Confidence: FIM determinantı büyükse model kendinden emin

Akış:
  1. Takımın/ligin geçmiş performans dağılımını parametrize et
  2. Fisher Information Matrix'i hesapla
  3. Mevcut maç dağılımı ile geçmiş arasındaki Fisher-Rao mesafesini ölç
  4. Mesafe büyükse → rejim değişimi / anomali
  5. Natural gradient ile model parametrelerini güncelle
  6. Tüm geometrik hesaplamalar loglanır

Teknoloji: numpy + scipy (tam analitik implementasyon)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from loguru import logger

try:
    from scipy import stats as sp_stats
    from scipy.integrate import quad
    from scipy.linalg import sqrtm, det, inv
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False


# ═══════════════════════════════════════════════
#  VERİ YAPILARI
# ═══════════════════════════════════════════════
@dataclass
class FisherReport:
    """Fisher Geometry analiz raporu."""
    match_id: str = ""
    # Fisher Information Matrix
    fim_determinant: float = 0.0     # det(I) — model güveni
    fim_trace: float = 0.0           # tr(I) — toplam bilgi
    fim_condition: float = 0.0       # cond(I) — sayısal kararlılık
    # Mesafeler
    fisher_rao_distance: float = 0.0  # Geodezik mesafe
    kl_divergence: float = 0.0       # KL(P||Q)
    jeffreys_divergence: float = 0.0 # Simetrik KL
    hellinger_distance: float = 0.0  # Hellinger mesafesi
    # Anomali
    is_anomaly: bool = False
    anomaly_score: float = 0.0
    # Rejim
    regime_shift: bool = False
    regime_confidence: float = 0.0
    # Tavsiye
    recommendation: str = ""
    method: str = "analytic"


# ═══════════════════════════════════════════════
#  FİSHER INFORMATION HESAPLAMALARI
# ═══════════════════════════════════════════════
def _fisher_normal(mu: float, sigma: float) -> np.ndarray:
    """Normal dağılım için analitik Fisher Information Matrix.

    I(μ,σ) = [[1/σ², 0], [0, 2/σ²]]
    """
    s2 = max(sigma ** 2, 1e-10)
    return np.array([
        [1.0 / s2, 0.0],
        [0.0, 2.0 / s2],
    ])


def _fisher_rao_normal(mu1: float, s1: float,
                         mu2: float, s2: float) -> float:
    """İki normal dağılım arasındaki Fisher-Rao geodezik mesafesi.

    Kapalı form: d_FR = √2 · |arcsinh((μ₁-μ₂)/(√2·σ)) - ...|
    Basitleştirilmiş: Rao distance for univariate normals.
    """
    s1 = max(s1, 1e-8)
    s2 = max(s2, 1e-8)

    # Fisher-Rao for 1D Gaussian (Poincaré half-plane metric)
    delta_mu = mu1 - mu2
    ratio = s1 / s2

    # Geodesic distance on the Poincaré half-plane
    d = np.sqrt(2) * np.arccosh(
        1 + (delta_mu ** 2 + (s1 - s2) ** 2) / (2 * s1 * s2)
    )
    return float(d)


def _kl_divergence_normal(mu1: float, s1: float,
                            mu2: float, s2: float) -> float:
    """KL(N(μ₁,σ₁) || N(μ₂,σ₂))."""
    s1 = max(s1, 1e-8)
    s2 = max(s2, 1e-8)
    return float(
        np.log(s2 / s1) + (s1 ** 2 + (mu1 - mu2) ** 2) / (2 * s2 ** 2) - 0.5
    )


def _hellinger_normal(mu1: float, s1: float,
                        mu2: float, s2: float) -> float:
    """Hellinger distance between two normals."""
    s1 = max(s1, 1e-8)
    s2 = max(s2, 1e-8)
    term1 = np.sqrt(2 * s1 * s2 / (s1 ** 2 + s2 ** 2))
    term2 = np.exp(-0.25 * (mu1 - mu2) ** 2 / (s1 ** 2 + s2 ** 2))
    return float(np.sqrt(1 - term1 * term2))


def _fisher_empirical(data: np.ndarray, n_params: int = 2) -> np.ndarray:
    """Ampirik Fisher Information Matrix (score fonksiyonu üzerinden).

    Log-likelihood gradyanlarının kovaryans matrisi.
    """
    n = len(data)
    if n < 10:
        return np.eye(n_params)

    mu = np.mean(data)
    sigma = max(np.std(data), 1e-8)

    # Score fonksiyonları (Normal dağılım varsayımı)
    score_mu = (data - mu) / (sigma ** 2)
    score_sigma = ((data - mu) ** 2 - sigma ** 2) / (sigma ** 3)

    scores = np.column_stack([score_mu, score_sigma])
    fim = scores.T @ scores / n

    return fim


# ═══════════════════════════════════════════════
#  FISHER GEOMETRY (Ana Sınıf)
# ═══════════════════════════════════════════════
class FisherGeometry:
    """Information Geometry motoru.

    Kullanım:
        fg = FisherGeometry(anomaly_threshold=2.0)

        # İki dağılım arası mesafe
        report = fg.compare_distributions(
            reference_data=historical_odds,
            current_data=live_odds,
            match_id="gs_fb",
        )

        if report.regime_shift:
            reduce_stakes()

        # Natural gradient
        grad = fg.natural_gradient(params, loss_grad)
    """

    def __init__(self, anomaly_threshold: float = 2.0,
                 regime_threshold: float = 1.5,
                 history_window: int = 100):
        self._anomaly_thresh = anomaly_threshold
        self._regime_thresh = regime_threshold
        self._history_win = history_window
        self._distance_history: list[float] = []

        logger.debug(
            f"[FisherGeo] Başlatıldı: anomaly_thresh={anomaly_threshold}, "
            f"regime_thresh={regime_threshold}"
        )

    def compare_distributions(self, reference_data: np.ndarray,
                                current_data: np.ndarray,
                                match_id: str = "") -> FisherReport:
        """İki veri seti arasındaki Fisher-Rao geometrik analizi."""
        report = FisherReport(match_id=match_id)

        ref = np.array(reference_data, dtype=np.float64).flatten()
        cur = np.array(current_data, dtype=np.float64).flatten()

        ref = ref[np.isfinite(ref)]
        cur = cur[np.isfinite(cur)]

        if len(ref) < 10 or len(cur) < 5:
            report.recommendation = "Yetersiz veri"
            return report

        # Parametreler
        mu_ref, s_ref = float(np.mean(ref)), float(np.std(ref))
        mu_cur, s_cur = float(np.mean(cur)), float(np.std(cur))
        s_ref = max(s_ref, 1e-8)
        s_cur = max(s_cur, 1e-8)

        # Fisher Information Matrix (referans)
        fim_ref = _fisher_normal(mu_ref, s_ref)
        report.fim_determinant = round(float(np.linalg.det(fim_ref)), 6)
        report.fim_trace = round(float(np.trace(fim_ref)), 6)
        report.fim_condition = round(
            float(np.linalg.cond(fim_ref)), 2,
        )

        # Fisher-Rao distance
        report.fisher_rao_distance = round(
            _fisher_rao_normal(mu_ref, s_ref, mu_cur, s_cur), 6,
        )

        # KL Divergence
        kl_fwd = _kl_divergence_normal(mu_cur, s_cur, mu_ref, s_ref)
        kl_rev = _kl_divergence_normal(mu_ref, s_ref, mu_cur, s_cur)
        report.kl_divergence = round(kl_fwd, 6)
        report.jeffreys_divergence = round(kl_fwd + kl_rev, 6)

        # Hellinger
        report.hellinger_distance = round(
            _hellinger_normal(mu_ref, s_ref, mu_cur, s_cur), 6,
        )

        # Anomali tespiti
        self._distance_history.append(report.fisher_rao_distance)
        if len(self._distance_history) > self._history_win:
            self._distance_history = self._distance_history[-self._history_win:]

        if len(self._distance_history) >= 10:
            mean_d = np.mean(self._distance_history[:-1])
            std_d = np.std(self._distance_history[:-1]) + 1e-8
            z_score = (report.fisher_rao_distance - mean_d) / std_d
            report.anomaly_score = round(float(z_score), 4)
            report.is_anomaly = z_score > self._anomaly_thresh
        else:
            report.anomaly_score = 0.0

        # Rejim değişimi
        report.regime_shift = report.fisher_rao_distance > self._regime_thresh
        report.regime_confidence = round(
            min(report.fisher_rao_distance / self._regime_thresh, 2.0), 4,
        )

        report.recommendation = self._advice(report)

        logger.debug(
            f"[FisherGeo] {match_id}: "
            f"FR={report.fisher_rao_distance:.4f}, "
            f"KL={report.kl_divergence:.4f}, "
            f"H={report.hellinger_distance:.4f}, "
            f"det(I)={report.fim_determinant:.4f}, "
            f"anomaly={report.is_anomaly}, "
            f"regime_shift={report.regime_shift}"
        )

        return report

    def natural_gradient(self, params: np.ndarray,
                           loss_gradient: np.ndarray,
                           data: np.ndarray | None = None) -> np.ndarray:
        """Natural gradient = F⁻¹ · ∇L.

        Fisher metriği ile ölçeklenmiş gradyan —
        parametre uzayında daha hızlı ve kararlı optimizasyon.
        """
        n = len(params)

        if data is not None and len(data) >= 10:
            fim = _fisher_empirical(data, n_params=n)
        else:
            fim = np.eye(n)

        try:
            fim_inv = np.linalg.inv(fim + np.eye(n) * 1e-6)
            nat_grad = fim_inv @ loss_gradient
        except np.linalg.LinAlgError:
            nat_grad = loss_gradient

        logger.debug(
            f"[FisherGeo] Natural gradient: "
            f"|∇L|={np.linalg.norm(loss_gradient):.4f}, "
            f"|F⁻¹∇L|={np.linalg.norm(nat_grad):.4f}"
        )

        return nat_grad

    def model_confidence(self, data: np.ndarray) -> float:
        """Model güven skoru = det(FIM)^(1/n).

        FIM determinantı büyükse model veriden çok bilgi çıkarıyor →
        yüksek güven. Küçükse → belirsizlik yüksek.
        """
        if len(data) < 10:
            return 0.0
        fim = _fisher_empirical(data)
        det_val = max(np.linalg.det(fim), 1e-20)
        confidence = float(det_val ** (1.0 / fim.shape[0]))
        return round(min(confidence, 1.0), 6)

    def _advice(self, r: FisherReport) -> str:
        if r.is_anomaly and r.regime_shift:
            return (
                f"KRİTİK: Anomali + rejim değişimi (FR={r.fisher_rao_distance:.3f}). "
                f"Dağılım geometrik olarak çok uzakta. Bahisleri durdur."
            )
        if r.regime_shift:
            return (
                f"UYARI: Rejim değişimi tespit (FR={r.fisher_rao_distance:.3f}). "
                f"Modeller güncellenmeli. Stake %50 düşür."
            )
        if r.is_anomaly:
            return (
                f"DİKKAT: Anomali skoru yüksek ({r.anomaly_score:.1f}σ). "
                f"Oran manipülasyonu olabilir. Dikkatli devam."
            )
        return (
            f"NORMAL: FR={r.fisher_rao_distance:.4f}, "
            f"det(I)={r.fim_determinant:.4f}. "
            f"Geometrik stabilite sağlanıyor."
        )
