"""
transport_metric.py – Optimal Transport (Wasserstein Distance).

"Bugünkü maç, geçmişteki X maçına benziyor" derken standart
uzaklık (Euclidean) yanıltır. İki olasılık dağılımını birbirine
dönüştürmenin "maliyetini" hesaplamalıyız.

Kavramlar:
  - Earth Mover's Distance (EMD): Bir dağılımı diğerine dönüştürmenin
    minimum "taşıma maliyeti"
  - Wasserstein-1: W₁(P,Q) = inf ∫|x-y| dγ(x,y)
  - Sinkhorn Distance: Regularize EMD (hızlı yaklaşık)
  - Optimal Transport Plan: Hangi "kütle" nereden nereye taşınıyor
  - Model Drift: Eğitim verisi ile canlı veri arasındaki kayma

Sinyaller:
  - W₁ < eşik → Dağılımlar benzer → Model güvenilir
  - W₁ > eşik → "Veri Rejimi Değişti" → BAHİS DURDUR
  - Drift Tespiti → Model yeniden eğitim tetikle

Teknoloji: pot (Python Optimal Transport Library)
Fallback: scipy.stats.wasserstein_distance
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from loguru import logger

try:
    import ot
    POT_OK = True
except ImportError:
    POT_OK = False
    logger.info("pot yüklü değil – scipy wasserstein fallback.")

try:
    from scipy.stats import wasserstein_distance as scipy_wasserstein
    from scipy.spatial.distance import cdist
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False


@dataclass
class TransportReport:
    """Optimal Transport raporu."""
    name: str = ""
    # Mesafeler
    wasserstein_1: float = 0.0          # 1D Wasserstein mesafesi
    wasserstein_2: float = 0.0          # 2D (çok boyutlu)
    sinkhorn: float = 0.0              # Regularize EMD
    max_mean_discrepancy: float = 0.0  # MMD (kernel bazlı)
    # Drift
    is_drifted: bool = False
    drift_severity: str = "none"       # none | mild | moderate | severe
    drift_dimensions: list[str] = field(default_factory=list)
    # Transport plan
    transport_cost: float = 0.0
    # Karar
    model_reliable: bool = True
    kill_betting: bool = False
    recommendation: str = ""
    method: str = ""


@dataclass
class DriftMonitor:
    """Sürekli drift izleme durumu."""
    reference_mean: np.ndarray | None = None
    reference_std: np.ndarray | None = None
    reference_dist: np.ndarray | None = None
    drift_count: int = 0
    check_count: int = 0
    last_wasserstein: float = 0.0
    rolling_distances: list[float] = field(default_factory=list)


# ═══════════════════════════════════════════════
#  WASSERSTEIN HESAPLAYICILAR
# ═══════════════════════════════════════════════
def wasserstein_1d(p: np.ndarray, q: np.ndarray) -> float:
    """1D Wasserstein mesafesi.

    İki dağılım arasındaki "Earth Mover's Distance".
    """
    if SCIPY_OK:
        return float(scipy_wasserstein(p, q))

    # Manuel: sıralı CDF farkının integrali
    p_sorted = np.sort(p)
    q_sorted = np.sort(q)

    n = max(len(p_sorted), len(q_sorted))
    p_interp = np.interp(
        np.linspace(0, 1, n),
        np.linspace(0, 1, len(p_sorted)),
        p_sorted,
    )
    q_interp = np.interp(
        np.linspace(0, 1, n),
        np.linspace(0, 1, len(q_sorted)),
        q_sorted,
    )

    return float(np.mean(np.abs(p_interp - q_interp)))


def wasserstein_nd(P: np.ndarray, Q: np.ndarray,
                    reg: float = 0.01) -> tuple[float, float]:
    """N-boyutlu Wasserstein ve Sinkhorn mesafeleri.

    Returns: (wasserstein, sinkhorn)
    """
    n_p = len(P)
    n_q = len(Q)

    if n_p == 0 or n_q == 0:
        return 0.0, 0.0

    # Eşit ağırlıklar
    a = np.ones(n_p) / n_p
    b = np.ones(n_q) / n_q

    if POT_OK:
        try:
            M = ot.dist(P, Q, metric="euclidean")

            # Exact Wasserstein (EMD)
            w_dist = float(ot.emd2(a, b, M))

            # Sinkhorn (hızlı yaklaşık)
            try:
                s_dist = float(ot.sinkhorn2(a, b, M, reg=reg)[0])
            except Exception as e:
                logger.debug(f"Exception caught: {e}")
                s_dist = w_dist

            return w_dist, s_dist
        except Exception as e:
            logger.debug(f"Exception caught: {e}")

    # Scipy fallback (sadece 1D)
    if SCIPY_OK and P.ndim == 1:
        w = float(scipy_wasserstein(P, Q))
        return w, w

    # Manuel: ortalama boyut-bazlı Wasserstein
    if P.ndim == 2 and Q.ndim == 2:
        total = 0.0
        for d in range(P.shape[1]):
            total += wasserstein_1d(P[:, d], Q[:, d])
        avg = total / P.shape[1]
        return avg, avg

    return 0.0, 0.0


def maximum_mean_discrepancy(P: np.ndarray, Q: np.ndarray,
                               gamma: float = 1.0) -> float:
    """Maximum Mean Discrepancy (MMD) – kernel bazlı dağılım farkı.

    Gaussian kernel: k(x,y) = exp(-γ||x-y||²)
    """
    if len(P) == 0 or len(Q) == 0:
        return 0.0

    P = P.reshape(-1, 1) if P.ndim == 1 else P
    Q = Q.reshape(-1, 1) if Q.ndim == 1 else Q

    def rbf_kernel(X, Y, g):
        if SCIPY_OK:
            dists = cdist(X, Y, "sqeuclidean")
        else:
            dists = np.sum((X[:, None] - Y[None, :]) ** 2, axis=-1)
        return np.exp(-g * dists)

    kpp = rbf_kernel(P, P, gamma)
    kqq = rbf_kernel(Q, Q, gamma)
    kpq = rbf_kernel(P, Q, gamma)

    mmd = float(np.mean(kpp) + np.mean(kqq) - 2 * np.mean(kpq))
    return max(0.0, mmd)


# ═══════════════════════════════════════════════
#  TRANSPORT METRIC (Ana Sınıf)
# ═══════════════════════════════════════════════
class TransportMetric:
    """Optimal Transport ile model drift tespiti.

    Kullanım:
        tm = TransportMetric()

        # Referans dağılım kaydet (eğitim verisi)
        tm.set_reference(X_train)

        # Canlı veri kontrolü
        report = tm.check_drift(X_live)
        if report.kill_betting:
            stop_all_bets()

        # İki dağılım karşılaştırma
        report = tm.compare(dist_A, dist_B)
    """

    # Drift eşikleri
    MILD_THRESHOLD = 0.10
    MODERATE_THRESHOLD = 0.25
    SEVERE_THRESHOLD = 0.50

    def __init__(self, drift_threshold: float = 0.25,
                 kill_threshold: float = 0.50,
                 sinkhorn_reg: float = 0.01):
        self._drift_threshold = drift_threshold
        self._kill_threshold = kill_threshold
        self._reg = sinkhorn_reg
        self._monitor = DriftMonitor()
        self._feature_names: list[str] = []
        logger.debug(
            f"[Transport] Metric başlatıldı "
            f"(drift={drift_threshold}, kill={kill_threshold})"
        )

    # ═══════════════════════════════════════════
    #  REFERANS DAĞILIM
    # ═══════════════════════════════════════════
    def set_reference(self, X: np.ndarray,
                       feature_names: list[str] | None = None) -> None:
        """Eğitim verisi dağılımını referans olarak kaydet."""
        X = np.array(X, dtype=np.float64)
        self._monitor.reference_dist = X
        self._monitor.reference_mean = np.mean(X, axis=0)
        self._monitor.reference_std = np.std(X, axis=0)
        self._feature_names = feature_names or [
            f"feat_{i}" for i in range(X.shape[1] if X.ndim == 2 else 1)
        ]
        logger.info(
            f"[Transport] Referans kaydedildi: "
            f"{X.shape[0]} örnek, {X.shape[1] if X.ndim == 2 else 1} boyut"
        )

    # ═══════════════════════════════════════════
    #  DRİFT KONTROLÜ
    # ═══════════════════════════════════════════
    def check_drift(self, X_live: np.ndarray,
                      name: str = "") -> TransportReport:
        """Canlı veri ile referans arasındaki drift'i ölç."""
        report = TransportReport(name=name or "drift_check")
        X_live = np.array(X_live, dtype=np.float64)

        if self._monitor.reference_dist is None:
            report.recommendation = "Referans dağılım yok – önce set_reference() çağırın."
            report.method = "no_reference"
            return report

        ref = self._monitor.reference_dist

        # 1D karşılaştırma (her özellik ayrı)
        if ref.ndim == 2 and X_live.ndim == 2:
            per_dim = []
            drift_dims = []
            for d in range(min(ref.shape[1], X_live.shape[1])):
                w = wasserstein_1d(ref[:, d], X_live[:, d])
                per_dim.append(w)
                if w > self._drift_threshold:
                    dim_name = (self._feature_names[d]
                                if d < len(self._feature_names)
                                else f"dim_{d}")
                    drift_dims.append(dim_name)
            report.wasserstein_1 = round(float(np.mean(per_dim)), 6)
            report.drift_dimensions = drift_dims

        # Çok boyutlu Wasserstein
        if ref.ndim >= 2 and X_live.ndim >= 2:
            # Performans: max 1000 örnek
            ref_sample = ref[:1000] if len(ref) > 1000 else ref
            live_sample = X_live[:1000] if len(X_live) > 1000 else X_live
            w2, sk = wasserstein_nd(ref_sample, live_sample, self._reg)
            report.wasserstein_2 = round(w2, 6)
            report.sinkhorn = round(sk, 6)

        # MMD
        report.max_mean_discrepancy = round(
            maximum_mean_discrepancy(
                ref[:500] if len(ref) > 500 else ref,
                X_live[:500] if len(X_live) > 500 else X_live,
            ), 6,
        )

        # Drift seviyesi
        w = max(report.wasserstein_1, report.wasserstein_2)
        if w < self.MILD_THRESHOLD:
            report.drift_severity = "none"
            report.model_reliable = True
        elif w < self.MODERATE_THRESHOLD:
            report.drift_severity = "mild"
            report.model_reliable = True
        elif w < self.SEVERE_THRESHOLD:
            report.drift_severity = "moderate"
            report.model_reliable = False
            report.is_drifted = True
        else:
            report.drift_severity = "severe"
            report.model_reliable = False
            report.is_drifted = True
            report.kill_betting = True

        # Monitor güncelle
        self._monitor.check_count += 1
        self._monitor.last_wasserstein = w
        self._monitor.rolling_distances.append(w)
        if len(self._monitor.rolling_distances) > 100:
            self._monitor.rolling_distances = self._monitor.rolling_distances[-100:]
        if report.is_drifted:
            self._monitor.drift_count += 1

        report.method = "pot" if POT_OK else "scipy"
        report.recommendation = self._generate_advice(report)

        return report

    # ═══════════════════════════════════════════
    #  İKİ DAĞILIM KARŞILAŞTIRMA
    # ═══════════════════════════════════════════
    def compare(self, P: np.ndarray, Q: np.ndarray,
                 name: str = "") -> TransportReport:
        """İki dağılımı doğrudan karşılaştır."""
        P = np.array(P, dtype=np.float64)
        Q = np.array(Q, dtype=np.float64)

        report = TransportReport(name=name or "compare")

        # 1D
        if P.ndim == 1 and Q.ndim == 1:
            report.wasserstein_1 = round(wasserstein_1d(P, Q), 6)

        # ND
        if P.ndim >= 1 and Q.ndim >= 1:
            P_2d = P.reshape(-1, 1) if P.ndim == 1 else P
            Q_2d = Q.reshape(-1, 1) if Q.ndim == 1 else Q
            w2, sk = wasserstein_nd(P_2d, Q_2d, self._reg)
            report.wasserstein_2 = round(w2, 6)
            report.sinkhorn = round(sk, 6)

        # MMD
        report.max_mean_discrepancy = round(
            maximum_mean_discrepancy(P, Q), 6,
        )

        report.method = "pot" if POT_OK else "scipy"
        w = max(report.wasserstein_1, report.wasserstein_2)
        if w < self.MILD_THRESHOLD:
            report.drift_severity = "none"
        elif w < self.MODERATE_THRESHOLD:
            report.drift_severity = "mild"
        elif w < self.SEVERE_THRESHOLD:
            report.drift_severity = "moderate"
            report.is_drifted = True
        else:
            report.drift_severity = "severe"
            report.is_drifted = True

        report.recommendation = self._generate_advice(report)
        return report

    # ═══════════════════════════════════════════
    #  FİLTRE
    # ═══════════════════════════════════════════
    def filter_drifted_bets(self, bets: list[dict],
                              live_features: np.ndarray | None = None
                              ) -> list[dict]:
        """Drift tespit edilmişse bahislere uyarı ekle."""
        if live_features is not None and self._monitor.reference_dist is not None:
            report = self.check_drift(live_features)
            if report.kill_betting:
                logger.warning(
                    f"[Transport] SEVERE DRIFT – tüm bahisler durduruldu! "
                    f"W={report.wasserstein_2:.4f}"
                )
                for bet in bets:
                    if isinstance(bet, dict):
                        bet["transport_killed"] = True
                        bet["wasserstein"] = report.wasserstein_2
                return []  # Tüm bahisleri iptal
            elif report.is_drifted:
                for bet in bets:
                    if isinstance(bet, dict):
                        bet["transport_drift"] = report.drift_severity
                        bet["wasserstein"] = report.wasserstein_2
                        # Moderate drift → stake düşür
                        original = bet.get("kelly_stake", bet.get("stake", 100))
                        bet["kelly_stake"] = round(original * 0.5, 2)
        return bets

    # ═══════════════════════════════════════════
    #  TAVSİYE
    # ═══════════════════════════════════════════
    def _generate_advice(self, report: TransportReport) -> str:
        """Drift tavsiyesi."""
        if report.kill_betting:
            return (
                f"SEVERE DRIFT: W={report.wasserstein_2:.4f}. "
                f"Veri rejimi değişti! Tüm bahisler DURDURULDU. "
                f"Model yeniden eğitim gerekli."
            )
        elif report.is_drifted:
            dims = ", ".join(report.drift_dimensions[:5]) if report.drift_dimensions else "?"
            return (
                f"MODERATE DRIFT: W={report.wasserstein_2:.4f}. "
                f"Kayma boyutları: [{dims}]. "
                f"Stake'ler %50 düşürüldü."
            )
        elif report.drift_severity == "mild":
            return (
                f"Hafif kayma: W={report.wasserstein_2:.4f}. "
                f"İzlemeye devam."
            )
        return f"Dağılımlar tutarlı (W={report.wasserstein_2:.4f}). Model güvenilir."

    @property
    def drift_rate(self) -> float:
        """Drift oranı (kontrol sayısına göre)."""
        if self._monitor.check_count == 0:
            return 0.0
        return self._monitor.drift_count / self._monitor.check_count
