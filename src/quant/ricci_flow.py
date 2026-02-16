"""
ricci_flow.py – Ricci Curvature (Diferansiyel Geometri).

Piyasa çökmeleri (Market Crash) aniden olmaz, ağın "Eğriliği"
(Curvature) değişir. Ricci Eğriliği, standart sapmanın
göremediği "Sistemik Riski" görür.

Kavramlar:
  - Ollivier-Ricci Curvature: İki düğüm arasındaki "geometrik mesafe"
    ile "ağ mesafesi" arasındaki fark
  - Pozitif eğrilik → ağ kararlı, düğümler birbirine yakın
  - Negatif eğrilik → ağ gergin, kopma/çökmeye yakın
  - Ricci Flow: Eğriliği zaman içinde izleyerek "toplu panik"
    veya "likidite krizi" tespiti

Sinyaller:
  - κ > 0: Ağ kararlı → Bahis güvenli
  - κ ≈ 0: Nötr → İzle
  - κ < -0.3: Gerginlik → Stake düşür
  - Δκ < -0.5 (ani düşüş): Panik → BAHİS DURDUR

Teknoloji: GraphRicciCurvature veya networkx + OT (pot)
Fallback: Manuel Ollivier-Ricci hesaplama
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from loguru import logger

try:
    import networkx as nx
    NX_OK = True
except ImportError:
    NX_OK = False

try:
    from GraphRicciCurvature.OllivierRicci import OllivierRicci
    RICCI_LIB_OK = True
except ImportError:
    RICCI_LIB_OK = False
    logger.debug("GraphRicciCurvature yüklü değil – manuel hesaplama.")

try:
    from scipy.optimize import linprog
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False


# ═══════════════════════════════════════════════
#  VERİ YAPILARI
# ═══════════════════════════════════════════════
@dataclass
class RicciReport:
    """Ricci eğrilik raporu."""
    name: str = ""
    # Global metrikler
    avg_curvature: float = 0.0
    min_curvature: float = 0.0
    max_curvature: float = 0.0
    std_curvature: float = 0.0
    # Temporal
    curvature_delta: float = 0.0      # Son ölçümden değişim
    curvature_trend: str = ""         # "stable" | "declining" | "rising"
    # Risk
    stress_level: str = "low"         # "low" | "moderate" | "high" | "critical"
    systemic_risk: float = 0.0       # 0–1 arası
    crisis_probability: float = 0.0   # Çöküş olasılığı
    # Kenar detayları
    critical_edges: list[tuple[str, str, float]] = field(default_factory=list)
    n_edges: int = 0
    n_negative: int = 0
    # Karar
    kill_betting: bool = False
    stake_multiplier: float = 1.0     # Risk çarpanı
    recommendation: str = ""
    method: str = ""


@dataclass
class CurvatureHistory:
    """Eğrilik tarihçesi (trend tespiti)."""
    timestamps: list[float] = field(default_factory=list)
    curvatures: list[float] = field(default_factory=list)


# ═══════════════════════════════════════════════
#  MANUEL OLLİVİER-RİCCİ HESAPLAMA
# ═══════════════════════════════════════════════
def _node_distribution(G: Any, node: Any, alpha: float = 0.5) -> dict:
    """Düğüm olasılık dağılımı (lazy random walk).

    P(v) = α · δ(v) + (1-α) · uniform(neighbors)
    """
    neighbors = list(G.neighbors(node))
    if not neighbors:
        return {node: 1.0}

    dist = {node: alpha}
    share = (1.0 - alpha) / len(neighbors)
    for nb in neighbors:
        dist[nb] = dist.get(nb, 0.0) + share

    return dist


def _wasserstein_1d_simple(mu: dict, nu: dict, G: Any) -> float:
    """Basitleştirilmiş 1-Wasserstein (ağ mesafesi üzerinden).

    Tam OT yerine, ortalama mesafe farkı yaklaşımı.
    """
    if not mu or not nu:
        return 0.0

    total = 0.0
    total_weight = 0.0

    for u, wu in mu.items():
        for v, wv in nu.items():
            try:
                if NX_OK:
                    d = nx.shortest_path_length(G, u, v)
                else:
                    d = 1  # Fallback
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                d = G.number_of_nodes()
            total += wu * wv * d
            total_weight += wu * wv

    return total / max(total_weight, 1e-15)


def ollivier_ricci_curvature(G: Any, u: Any, v: Any,
                               alpha: float = 0.5) -> float:
    """Ollivier-Ricci eğriliği hesapla.

    κ(u,v) = 1 - W₁(μ_u, μ_v) / d(u,v)

    Pozitif → düğümler yakın (kararlı)
    Negatif → düğümler uzak (gergin)
    """
    mu_u = _node_distribution(G, u, alpha)
    mu_v = _node_distribution(G, v, alpha)

    # Ağ mesafesi
    try:
        d_uv = nx.shortest_path_length(G, u, v)
    except Exception:
        d_uv = 1

    if d_uv == 0:
        return 0.0

    # Wasserstein mesafesi
    w1 = _wasserstein_1d_simple(mu_u, mu_v, G)

    curvature = 1.0 - w1 / max(d_uv, 1e-15)
    return float(curvature)


def compute_all_curvatures(G: Any, alpha: float = 0.5
                            ) -> dict[tuple, float]:
    """Tüm kenarların Ricci eğriliğini hesapla."""
    curvatures = {}
    for u, v in G.edges():
        curvatures[(u, v)] = ollivier_ricci_curvature(G, u, v, alpha)
    return curvatures


# ═══════════════════════════════════════════════
#  RICCI FLOW ANALYZER (Ana Sınıf)
# ═══════════════════════════════════════════════
class RicciFlowAnalyzer:
    """Diferansiyel geometri ile piyasa riski analizi.

    Kullanım:
        rfa = RicciFlowAnalyzer()

        # Piyasa ağı oluştur (takımlar + oranlar arası ilişkiler)
        G = nx.Graph()
        G.add_edge("GS", "FB", weight=0.8)
        G.add_edge("GS", "BJK", weight=0.5)
        G.add_edge("FB", "BJK", weight=0.6)

        report = rfa.analyze(G, name="super_lig_week_20")
        if report.kill_betting:
            stop_all()
    """

    # Eşikler
    MODERATE_THRESHOLD = -0.1
    HIGH_THRESHOLD = -0.3
    CRITICAL_THRESHOLD = -0.5
    DELTA_PANIC = -0.5          # Ani düşüş eşiği

    def __init__(self, alpha: float = 0.5,
                 history_size: int = 50):
        self._alpha = alpha
        self._history = CurvatureHistory()
        self._history_size = history_size
        logger.debug(f"[Ricci] FlowAnalyzer başlatıldı (alpha={alpha})")

    def analyze(self, G: Any, name: str = "") -> RicciReport:
        """Ağın Ricci eğriliğini analiz et."""
        report = RicciReport(name=name or "ricci_analysis")

        if not NX_OK:
            report.recommendation = "networkx yüklü değil."
            report.method = "none"
            return report

        if G.number_of_edges() == 0:
            report.recommendation = "Ağda kenar yok."
            return report

        # Eğrilikleri hesapla
        if RICCI_LIB_OK:
            try:
                orc = OllivierRicci(G, alpha=self._alpha)
                orc.compute_ricci_curvature()
                curvatures = {
                    (u, v): G[u][v].get("ricciCurvature", 0.0)
                    for u, v in G.edges()
                }
                report.method = "GraphRicciCurvature"
            except Exception:
                curvatures = compute_all_curvatures(G, self._alpha)
                report.method = "manual_ollivier"
        else:
            curvatures = compute_all_curvatures(G, self._alpha)
            report.method = "manual_ollivier"

        if not curvatures:
            report.recommendation = "Eğrilik hesaplanamadı."
            return report

        # İstatistikler
        values = list(curvatures.values())
        report.n_edges = len(values)
        report.avg_curvature = round(float(np.mean(values)), 6)
        report.min_curvature = round(float(np.min(values)), 6)
        report.max_curvature = round(float(np.max(values)), 6)
        report.std_curvature = round(float(np.std(values)), 6)
        report.n_negative = sum(1 for v in values if v < 0)

        # Kritik kenarlar (en negatif)
        sorted_edges = sorted(curvatures.items(), key=lambda x: x[1])
        report.critical_edges = [
            (str(u), str(v), round(c, 4))
            for (u, v), c in sorted_edges[:5]
        ]

        # Temporal trend
        self._history.timestamps.append(time.time())
        self._history.curvatures.append(report.avg_curvature)
        if len(self._history.curvatures) > self._history_size:
            self._history.timestamps = self._history.timestamps[-self._history_size:]
            self._history.curvatures = self._history.curvatures[-self._history_size:]

        if len(self._history.curvatures) >= 2:
            report.curvature_delta = round(
                self._history.curvatures[-1] - self._history.curvatures[-2], 6,
            )
            if report.curvature_delta < self.DELTA_PANIC:
                report.curvature_trend = "panic_decline"
            elif report.curvature_delta < -0.1:
                report.curvature_trend = "declining"
            elif report.curvature_delta > 0.1:
                report.curvature_trend = "rising"
            else:
                report.curvature_trend = "stable"
        else:
            report.curvature_trend = "insufficient_data"

        # Stress seviyesi
        avg = report.avg_curvature
        if avg < self.CRITICAL_THRESHOLD:
            report.stress_level = "critical"
            report.systemic_risk = 0.9
            report.crisis_probability = 0.7
            report.kill_betting = True
            report.stake_multiplier = 0.0
        elif avg < self.HIGH_THRESHOLD:
            report.stress_level = "high"
            report.systemic_risk = 0.6
            report.crisis_probability = 0.4
            report.stake_multiplier = 0.3
        elif avg < self.MODERATE_THRESHOLD:
            report.stress_level = "moderate"
            report.systemic_risk = 0.3
            report.crisis_probability = 0.15
            report.stake_multiplier = 0.7
        else:
            report.stress_level = "low"
            report.systemic_risk = max(0, -avg * 0.5)
            report.crisis_probability = max(0, -avg * 0.2)
            report.stake_multiplier = 1.0

        # Ani düşüş kontrolü
        if report.curvature_delta < self.DELTA_PANIC:
            report.kill_betting = True
            report.stake_multiplier = 0.0
            report.stress_level = "critical"

        report.recommendation = self._advice(report)
        return report

    def build_market_graph(self, matches: list[dict],
                             odds_field: str = "home_odds") -> Any:
        """Maç listesinden piyasa ağı oluştur.

        Düğümler: Takımlar
        Kenarlar: Maç ilişkileri (oran benzerliği ağırlığıyla)
        """
        if not NX_OK:
            return None

        G = nx.Graph()

        for match in matches:
            home = match.get("home_team", "")
            away = match.get("away_team", "")
            if not home or not away:
                continue

            odds_h = match.get("home_odds", 2.0)
            odds_a = match.get("away_odds", 2.0)

            # Ağırlık: oranlar arasındaki ters mesafe
            weight = 1.0 / (1.0 + abs(odds_h - odds_a))
            G.add_edge(home, away, weight=weight)

        return G

    def adjust_bets_by_curvature(self, bets: list[dict],
                                   curvature_report: RicciReport
                                   ) -> list[dict]:
        """Eğriliğe göre bahis stake'lerini ayarla."""
        if curvature_report.kill_betting:
            logger.warning("[Ricci] CRITICAL – tüm bahisler iptal!")
            for bet in bets:
                if isinstance(bet, dict):
                    bet["ricci_killed"] = True
            return []

        mult = curvature_report.stake_multiplier
        if mult < 1.0:
            for bet in bets:
                if isinstance(bet, dict):
                    original = bet.get("kelly_stake", bet.get("stake", 100))
                    bet["kelly_stake"] = round(original * mult, 2)
                    bet["ricci_stress"] = curvature_report.stress_level
                    bet["ricci_curvature"] = curvature_report.avg_curvature

        return bets

    def _advice(self, r: RicciReport) -> str:
        if r.kill_betting:
            return (
                f"KRİTİK: Ağ eğriliği κ={r.avg_curvature:.4f} "
                f"(Δ={r.curvature_delta:+.4f}). "
                f"SİSTEMİK RİSK! Tüm bahisler durduruldu."
            )
        if r.stress_level == "high":
            return (
                f"Yüksek stres: κ={r.avg_curvature:.4f}, "
                f"kriz olasılığı={r.crisis_probability:.0%}. "
                f"Stake x{r.stake_multiplier:.1f}."
            )
        if r.stress_level == "moderate":
            return (
                f"Orta stres: κ={r.avg_curvature:.4f}. "
                f"İzlemeye devam, stake x{r.stake_multiplier:.1f}."
            )
        return (
            f"Ağ kararlı: κ={r.avg_curvature:.4f}, "
            f"negatif kenar={r.n_negative}/{r.n_edges}."
        )
