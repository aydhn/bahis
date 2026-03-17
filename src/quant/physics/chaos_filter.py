"""
chaos_filter.py – Lyapunov Exponents (Kaos Ölçümü).

Piyasa bazen tahmin edilebilir (Stabil), bazen tamamen
rastgeledir (Kaotik). Standart sapma bunu ayırt edemez.
Lyapunov Üssü ayırır.

Kavramlar:
  - Lyapunov Exponent (λ): Başlangıç koşullarına hassasiyetin ölçüsü
    λ < 0 → Stabil (Bahis Oyna)
    λ ≈ 0 → Sınırda (Dikkatli Ol)
    λ > 0 → Kaotik (Bahis Oynama!)
  - Kelebek Etkisi: Küçük bir değişiklik sonucu tamamen değiştirir
  - Correlation Dimension: Çekicinin (Attractor) boyutu
  - Sample Entropy: Zaman serisinin karmaşıklık ölçüsü
  - Recurrence Quantification (RQA): Tekrarlanma analizi

Sinyal Mantığı:
  λ > 0.05 → Kaotik → Tüm modeller "Kazanır" dese bile iptal et
  λ < -0.05 → Stabil → Modellere güven artır
  0.01 < λ < 0.05 → Sınırda → Stake'i %50 düşür

Teknoloji: nolds (Nonlinear Measures for Dynamical Systems)
Fallback: Manuel Lyapunov tahmini (Rosenstein yöntemi)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from loguru import logger

try:
    import nolds
    NOLDS_OK = True
except (ImportError, TypeError):
    NOLDS_OK = False
    logger.debug("nolds yüklü değil – manuel Lyapunov fallback.")

# ═══════════════════════════════════════════════
#  VERİ YAPILARI
# ═══════════════════════════════════════════════
@dataclass
class ChaosParams:
    """Kaos parametreleri."""
    max_lyapunov: float = 0.0       # En büyük Lyapunov üssü
    correlation_dim: float = 0.0    # Korelasyon boyutu
    sample_entropy: float = 0.0     # Örneklem entropisi
    hurst_exponent: float = 0.5     # Hurst üssü (referans)
    dfa: float = 0.0                # Detrended Fluctuation Analysis


@dataclass
class ChaosReport:
    """Kaos analizi raporu."""
    match_id: str = ""
    market: str = ""             # "home_odds" | "over_under" vs.
    # Parametreler
    params: ChaosParams = field(default_factory=ChaosParams)
    # Rejim
    regime: str = "unknown"      # "stable" | "edge_of_chaos" | "chaotic"
    chaos_score: float = 0.0     # 0-1 arası kaos skoru
    predictability: float = 1.0  # 0-1 arası tahmin edilebilirlik
    # Sinyaller
    kill_betting: bool = False   # Tüm bahisleri iptal et
    reduce_stake: bool = False   # Stake'i %50 düşür
    boost_confidence: bool = False  # Modellere güveni artır
    # Detay
    n_observations: int = 0
    method: str = ""
    recommendation: str = ""


# ═══════════════════════════════════════════════
#  MANUEL LYAPUNOV TAHMİNİ (Rosenstein Yöntemi)
# ═══════════════════════════════════════════════
def rosenstein_lyapunov(data: np.ndarray, emb_dim: int = 3,
                          lag: int = 1,
                          min_tsep: int | None = None) -> float:
    """Rosenstein yöntemiyle en büyük Lyapunov üssü.

    Adımlar:
    1. Zaman gecikmeli gömme (Time-delay embedding)
    2. Her nokta için en yakın komşuyu bul (zamansal olarak ayrık)
    3. Komşu çiftlerin zamanla nasıl uzaklaştığını ölç
    4. log(uzaklık) vs. zaman eğimi → λ_max
    """
    n = len(data)
    if n < emb_dim * lag + 10:
        return 0.0

    if min_tsep is None:
        min_tsep = emb_dim * lag

    # Zaman gecikmeli gömme
    m = n - (emb_dim - 1) * lag
    embedded = np.zeros((m, emb_dim))
    for i in range(emb_dim):
        embedded[:, i] = data[i * lag: i * lag + m]

    # Her nokta için en yakın komşuyu bul
    divergence = []
    max_iter = min(m // 2, 20)

    for i in range(m - max_iter):
        min_dist = float("inf")
        min_j = -1
        for j in range(m - max_iter):
            if abs(i - j) < min_tsep:
                continue
            dist = np.sqrt(np.sum((embedded[i] - embedded[j]) ** 2))
            if dist < min_dist and dist > 0:
                min_dist = dist
                min_j = j

        if min_j >= 0:
            # Uzaklaşma eğrisi
            for k in range(max_iter):
                if i + k < m and min_j + k < m:
                    d = np.sqrt(
                        np.sum((embedded[i + k] - embedded[min_j + k]) ** 2)
                    )
                    if d > 0:
                        while len(divergence) <= k:
                            divergence.append([])
                        divergence[k].append(np.log(d))

    if not divergence or len(divergence) < 3:
        return 0.0

    # Ortalama log-uzaklık
    avg_divergence = []
    for k, vals in enumerate(divergence):
        if vals:
            avg_divergence.append(np.mean(vals))

    if len(avg_divergence) < 3:
        return 0.0

    # Eğim (en küçük kareler)
    x = np.arange(len(avg_divergence), dtype=np.float64)
    y = np.array(avg_divergence)
    n_pts = len(x)
    slope = (n_pts * np.sum(x * y) - np.sum(x) * np.sum(y)) / (
        n_pts * np.sum(x**2) - np.sum(x) ** 2 + 1e-10
    )

    return float(slope)


def sample_entropy_manual(data: np.ndarray, m: int = 2,
                            r: float | None = None) -> float:
    """Manuel Sample Entropy hesaplama.

    SampEn(m, r) = -ln(A/B)
    A: m+1 uzunluğundaki benzer çift sayısı
    B: m uzunluğundaki benzer çift sayısı
    """
    n = len(data)
    if n < m + 2:
        return 0.0

    if r is None:
        r = 0.2 * np.std(data)
    if r <= 0:
        return 0.0

    def count_matches(template_len: int) -> int:
        count = 0
        templates = np.array([
            data[i:i + template_len] for i in range(n - template_len)
        ])
        for i in range(len(templates)):
            for j in range(i + 1, len(templates)):
                if np.max(np.abs(templates[i] - templates[j])) < r:
                    count += 1
        return count

    B = count_matches(m)
    A = count_matches(m + 1)

    if B == 0 or A == 0:
        return 0.0

    return -np.log(A / B)


# ═══════════════════════════════════════════════
#  CHAOS FILTER (Ana Sınıf)
# ═══════════════════════════════════════════════
class ChaosFilter:
    """Kaos teorisi ile tahmin edilemezlik filtresi.

    Kullanım:
        cf = ChaosFilter()

        # Oran geçmişi (zaman serisi)
        odds_history = [1.85, 1.82, 1.88, 1.90, 1.75, ...]

        report = cf.analyze(
            odds_history,
            match_id="gs_fb",
            market="home_odds",
        )

        if report.kill_betting:
            cancel_all_bets()
        elif report.reduce_stake:
            reduce_kelly_by_50%()
    """

    CHAOS_THRESHOLD = 0.05      # λ > bu → kaotik
    STABLE_THRESHOLD = -0.05    # λ < bu → stabil
    EDGE_UPPER = 0.05
    EDGE_LOWER = 0.01

    def __init__(self, emb_dim: int = 3, lag: int = 1):
        self._emb_dim = emb_dim
        self._lag = lag
        logger.debug(
            f"[Chaos] Filter başlatıldı: emb_dim={emb_dim}, lag={lag}"
        )

    def analyze(self, time_series: list[float] | np.ndarray,
                  match_id: str = "",
                  market: str = "odds") -> ChaosReport:
        """Zaman serisini kaos analizi ile filtrele."""
        report = ChaosReport(match_id=match_id, market=market)
        data = np.array(time_series, dtype=np.float64).flatten()
        report.n_observations = len(data)

        if len(data) < 10:
            report.regime = "insufficient_data"
            report.recommendation = "Yeterli veri yok (min 10 gözlem)."
            report.method = "none"
            return report

        params = ChaosParams()

        # Lyapunov üssü
        if NOLDS_OK:
            try:
                params.max_lyapunov = round(float(
                    nolds.lyap_r(data, emb_dim=self._emb_dim, lag=self._lag)
                ), 6)
                report.method = "nolds"
            except Exception as e:
                logger.debug(f"Exception caught: {e}")
                params.max_lyapunov = round(
                    rosenstein_lyapunov(data, self._emb_dim, self._lag), 6,
                )
                report.method = "rosenstein_manual"
        else:
            params.max_lyapunov = round(
                rosenstein_lyapunov(data, self._emb_dim, self._lag), 6,
            )
            report.method = "rosenstein_manual"

        # Korelasyon boyutu
        if NOLDS_OK:
            try:
                params.correlation_dim = round(float(
                    nolds.corr_dim(data, emb_dim=self._emb_dim)
                ), 4)
            except Exception as e:
                logger.debug(f"Exception caught: {e}")
                params.correlation_dim = 0.0
        else:
            params.correlation_dim = 0.0

        # Sample entropy
        if NOLDS_OK:
            try:
                params.sample_entropy = round(float(
                    nolds.sampen(data, emb_dim=2)
                ), 4)
            except Exception as e:
                logger.debug(f"Exception caught: {e}")
                params.sample_entropy = round(
                    sample_entropy_manual(data), 4,
                )
        else:
            params.sample_entropy = round(
                sample_entropy_manual(data), 4,
            )

        # Hurst üssü
        if NOLDS_OK:
            try:
                params.hurst_exponent = round(float(
                    nolds.hurst_rs(data)
                ), 4)
            except Exception as e:
                logger.debug(f"Exception caught: {e}")
                params.hurst_exponent = 0.5

        # DFA
        if NOLDS_OK:
            try:
                params.dfa = round(float(nolds.dfa(data)), 4)
            except Exception as e:
                logger.debug(f"Exception caught: {e}")
                params.dfa = 0.5

        report.params = params

        # Rejim belirleme
        lam = params.max_lyapunov
        if lam > self.CHAOS_THRESHOLD:
            report.regime = "chaotic"
            report.chaos_score = round(
                min(1.0, lam / (self.CHAOS_THRESHOLD * 5)), 3,
            )
            report.predictability = round(
                max(0, 1 - report.chaos_score), 3,
            )
        elif lam < self.STABLE_THRESHOLD:
            report.regime = "stable"
            report.chaos_score = 0.0
            report.predictability = round(
                min(1.0, abs(lam) / 0.2 + 0.5), 3,
            )
        elif self.EDGE_LOWER < lam < self.EDGE_UPPER:
            report.regime = "edge_of_chaos"
            report.chaos_score = round(lam / self.EDGE_UPPER, 3)
            report.predictability = round(0.5, 3)
        else:
            report.regime = "stable"
            report.chaos_score = round(max(0, lam / self.CHAOS_THRESHOLD), 3)
            report.predictability = round(
                1 - report.chaos_score * 0.5, 3,
            )

        # Sinyaller
        if report.regime == "chaotic":
            report.kill_betting = True
        elif report.regime == "edge_of_chaos":
            report.reduce_stake = True
        elif report.regime == "stable" and report.predictability > 0.7:
            report.boost_confidence = True

        report.recommendation = self._advice(report)
        return report

    def filter_bets(self, bets: list[dict],
                      odds_histories: dict[str, list[float]]) -> list[dict]:
        """Kaotik maçları bahis listesinden filtrele."""
        filtered = []
        for bet in bets:
            if not isinstance(bet, dict):
                filtered.append(bet)
                continue

            mid = bet.get("match_id", "")
            history = odds_histories.get(mid, [])

            if len(history) < 10:
                filtered.append(bet)
                continue

            report = self.analyze(history, match_id=mid)

            if report.kill_betting:
                bet["chaos_killed"] = True
                bet["chaos_regime"] = "chaotic"
                bet["chaos_lyapunov"] = report.params.max_lyapunov
                logger.warning(
                    f"[Chaos] {mid}: KAOS! λ={report.params.max_lyapunov:.4f} "
                    f"→ bahis iptal!"
                )
            elif report.reduce_stake:
                bet["chaos_regime"] = "edge_of_chaos"
                bet["chaos_stake_mult"] = 0.5
                old_stake = bet.get("stake", 0)
                bet["stake"] = old_stake * 0.5
            elif report.boost_confidence:
                bet["chaos_regime"] = "stable"
                bet["chaos_boost"] = True

            filtered.append(bet)

        return filtered

    def _advice(self, r: ChaosReport) -> str:
        if r.kill_betting:
            return (
                f"KAOS: λ={r.params.max_lyapunov:.4f} > 0 – "
                f"Kelebek Etkisi aktif! Tahmin edilemez. "
                f"TÜM BAHİSLER İPTAL."
            )
        if r.reduce_stake:
            return (
                f"SINIRDA: λ={r.params.max_lyapunov:.4f}, "
                f"kaos skoru={r.chaos_score:.0%}. "
                f"Stake %50 düşürüldü."
            )
        if r.boost_confidence:
            return (
                f"STABİL: λ={r.params.max_lyapunov:.4f} < 0, "
                f"tahmin edilebilirlik={r.predictability:.0%}. "
                f"Model güvenilir."
            )
        return (
            f"Rejim: {r.regime}, λ={r.params.max_lyapunov:.4f}, "
            f"entropi={r.params.sample_entropy:.3f}."
        )
