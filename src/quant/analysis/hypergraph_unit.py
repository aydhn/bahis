"""
hypergraph_unit.py – Hypergraph Neural Networks (Hiper-Çizge Ağları).

Klasik grafiklerde bir kenar (Edge) sadece 2 düğümü bağlar.
Futbol "Ünitelerle" oynanır: Defans Hattı (4 kişi), Orta Saha
Üçlüsü (3 kişi), Hücum Bloğu (3 kişi).

Hiper-Kenar (Hyperedge): Bir kenar N düğümü aynı anda bağlar.
→ "Defans hattındaki 4 oyuncudan biri hata yaptığında, tüm
   hattın çökme olasılığı nedir?" sorusunu çözer.

Kavramlar:
  - Hypergraph Incidence Matrix (H): Düğüm × Hiper-kenar
  - Vertex Degree: Bir düğümün kaç hiper-kenara dahil olduğu
  - Hyperedge Weight: Taktiksel birimin gücü
  - Laplacian: D_v^{-1/2} H W D_e^{-1} H^T D_v^{-1/2}
  - Unit Failure Probability: Birim çöküş olasılığı

Teknoloji: dhg (DeepHypergraph) veya torch_geometric
Fallback: numpy + scipy (Laplacian hesaplama)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from loguru import logger

try:
    import torch
    import torch.nn as nn
    TORCH_OK = True
except ImportError:
    TORCH_OK = False

try:
    from scipy.sparse import csr_matrix, diags
    from scipy.sparse.linalg import eigsh
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False


# ═══════════════════════════════════════════════
#  VERİ YAPILARI
# ═══════════════════════════════════════════════
@dataclass
class TacticalUnit:
    """Taktiksel birim (Hiper-kenar)."""
    name: str = ""                      # "Defans Hattı", "Orta Saha Üçlüsü"
    unit_type: str = ""                 # "defense", "midfield", "attack", "set_piece"
    player_indices: list[int] = field(default_factory=list)
    player_names: list[str] = field(default_factory=list)
    weight: float = 1.0                # Birimin önem ağırlığı
    cohesion: float = 0.0              # Birim uyumu (0–1)


@dataclass
class UnitAnalysis:
    """Birim analiz raporu."""
    unit: TacticalUnit = field(default_factory=TacticalUnit)
    failure_prob: float = 0.0           # Birim çöküş olasılığı
    weakest_link: str = ""              # En zayıf halka
    weakest_link_impact: float = 0.0    # Zayıf halka etkisi
    centrality_score: float = 0.0       # Birim merkeziliği
    redundancy: float = 0.0             # Yedeklilik (0=kritik, 1=yedekli)


@dataclass
class HypergraphReport:
    """Takım hiper-çizge raporu."""
    team: str = ""
    n_players: int = 0
    n_units: int = 0
    # Birim analizleri
    unit_analyses: list[UnitAnalysis] = field(default_factory=list)
    # Global metrikler
    team_cohesion: float = 0.0          # Takım uyumu (0–1)
    structural_entropy: float = 0.0     # Yapısal entropi
    vulnerability_index: float = 0.0    # Kırılganlık endeksi
    # Sinyaller
    defense_alert: bool = False
    midfield_alert: bool = False
    attack_alert: bool = False
    recommendation: str = ""


# ═══════════════════════════════════════════════
#  HYPERGRAPH İŞLEMLERİ
# ═══════════════════════════════════════════════
def build_incidence_matrix(n_nodes: int,
                            hyperedges: list[list[int]],
                            weights: list[float] | None = None
                            ) -> tuple[np.ndarray, np.ndarray]:
    """Hiper-çizge Incidence Matrisi H (n_nodes × n_edges) oluştur.

    Returns: (H, edge_weights)
    """
    n_edges = len(hyperedges)
    H = np.zeros((n_nodes, n_edges), dtype=np.float64)

    for e_idx, edge in enumerate(hyperedges):
        for node_idx in edge:
            if 0 <= node_idx < n_nodes:
                H[node_idx, e_idx] = 1.0

    if weights is None:
        weights = np.ones(n_edges)
    else:
        weights = np.array(weights, dtype=np.float64)

    return H, weights


def hypergraph_laplacian(H: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Hiper-çizge Laplacian'ı hesapla.

    L = D_v^{-1/2} · H · W · D_e^{-1} · H^T · D_v^{-1/2}

    D_v: Düğüm derece matrisi
    D_e: Hiper-kenar derece matrisi
    """
    # Düğüm dereceleri
    D_v = np.sum(H, axis=1)  # Her düğümün kaç hiper-kenara dahil olduğu
    D_v_inv_sqrt = np.where(D_v > 0, 1.0 / np.sqrt(D_v), 0.0)

    # Hiper-kenar dereceleri
    D_e = np.sum(H, axis=0)  # Her hiper-kenardaki düğüm sayısı
    D_e_inv = np.where(D_e > 0, 1.0 / D_e, 0.0)

    # Laplacian: D_v^{-1/2} H W D_e^{-1} H^T D_v^{-1/2}
    Dv = np.diag(D_v_inv_sqrt)
    De = np.diag(D_e_inv)
    Wm = np.diag(W)

    L = Dv @ H @ Wm @ De @ H.T @ Dv
    return L


def unit_failure_probability(H: np.ndarray, player_ratings: np.ndarray,
                               edge_idx: int,
                               missing_player: int | None = None
                               ) -> float:
    """Taktiksel birimin çöküş olasılığını hesapla.

    Eğer bir oyuncu eksikse (sakatlık/kırmızı kart),
    birimin performans kaybını modeller.
    """
    edge_players = np.where(H[:, edge_idx] > 0)[0]
    if len(edge_players) == 0:
        return 1.0

    ratings = player_ratings[edge_players]
    if missing_player is not None and missing_player in edge_players:
        # Eksik oyuncunun katkısını çıkar
        mask = edge_players != missing_player
        ratings = ratings[mask] if mask.any() else np.array([30.0])

    # Birim gücü: harmonik ortalama (en zayıf halkaya duyarlı)
    if len(ratings) == 0 or np.any(ratings <= 0):
        return 1.0

    harmonic_mean = len(ratings) / np.sum(1.0 / np.maximum(ratings, 1.0))

    # Çöküş olasılığı: sigmoid terslemesi
    failure = 1.0 / (1.0 + np.exp((harmonic_mean - 60) / 10))
    return float(np.clip(failure, 0.0, 1.0))


# ═══════════════════════════════════════════════
#  HYPERGRAPH UNIT ANALYZER (Ana Sınıf)
# ═══════════════════════════════════════════════
class HypergraphUnitAnalyzer:
    """Taktiksel birim analizi (Hiper-Çizge tabanlı).

    Kullanım:
        hg = HypergraphUnitAnalyzer()

        # Takım tanımla
        units = [
            TacticalUnit("Defans Hattı", "defense", [0,1,2,3], weight=1.5),
            TacticalUnit("Orta Saha Üçlüsü", "midfield", [4,5,6], weight=1.2),
            TacticalUnit("Hücum Bloğu", "attack", [7,8,9,10], weight=1.0),
        ]
        ratings = np.array([75, 80, 72, 78, 85, 70, 82, 90, 65, 77, 88])

        report = hg.analyze_team("Galatasaray", units, ratings)
        if report.defense_alert:
            reduce_bet_confidence()
    """

    FAILURE_THRESHOLD = 0.35     # Birim çöküş uyarı eşiği
    COHESION_THRESHOLD = 0.40    # Düşük uyum eşiği

    def __init__(self, failure_threshold: float = 0.35):
        self.FAILURE_THRESHOLD = failure_threshold
        logger.debug("[Hypergraph] UnitAnalyzer başlatıldı.")

    def analyze_team(self, team: str,
                      units: list[TacticalUnit],
                      player_ratings: np.ndarray,
                      missing_players: list[int] | None = None
                      ) -> HypergraphReport:
        """Takımın taktiksel birimlerini hiper-çizge ile analiz et."""
        report = HypergraphReport(team=team)
        player_ratings = np.array(player_ratings, dtype=np.float64)
        n_players = len(player_ratings)
        report.n_players = n_players
        report.n_units = len(units)

        if not units or n_players == 0:
            report.recommendation = "Veri yetersiz."
            return report

        missing = set(missing_players or [])

        # Incidence matrix
        hyperedges = [u.player_indices for u in units]
        weights = [u.weight for u in units]
        H, W = build_incidence_matrix(n_players, hyperedges, weights)

        # Laplacian
        try:
            L = hypergraph_laplacian(H, W)
            # Yapısal entropi (eigenvalue bazlı)
            eigvals = np.linalg.eigvalsh(L)
            pos_eigvals = eigvals[eigvals > 1e-10]
            if len(pos_eigvals) > 0:
                norm_eig = pos_eigvals / pos_eigvals.sum()
                report.structural_entropy = float(
                    -np.sum(norm_eig * np.log2(norm_eig + 1e-15))
                )
        except Exception:
            pass

        # Birim bazlı analiz
        total_failure = 0.0
        for e_idx, unit in enumerate(units):
            ua = UnitAnalysis(unit=unit)

            # Normal çöküş olasılığı
            base_failure = unit_failure_probability(
                H, player_ratings, e_idx,
            )

            # Eksik oyuncu etkisi
            worst_impact = 0.0
            worst_player = ""
            for pidx in unit.player_indices:
                if pidx in missing:
                    f_without = unit_failure_probability(
                        H, player_ratings, e_idx, missing_player=pidx,
                    )
                    impact = f_without - base_failure
                    if impact > worst_impact:
                        worst_impact = impact
                        worst_player = (
                            unit.player_names[unit.player_indices.index(pidx)]
                            if pidx in unit.player_indices and
                               unit.player_indices.index(pidx) < len(unit.player_names)
                            else f"player_{pidx}"
                        )

            # Eğer eksik oyuncu varsa, gerçek çöküş olasılığı güncelle
            if missing:
                for pidx in unit.player_indices:
                    if pidx in missing:
                        base_failure = unit_failure_probability(
                            H, player_ratings, e_idx, missing_player=pidx,
                        )
                        break

            ua.failure_prob = round(base_failure, 4)
            ua.weakest_link = worst_player
            ua.weakest_link_impact = round(worst_impact, 4)

            # Birim uyumu: rating standart sapmasının tersi
            unit_ratings = player_ratings[
                [i for i in unit.player_indices if 0 <= i < n_players]
            ]
            if len(unit_ratings) > 1:
                std = float(np.std(unit_ratings))
                ua.cohesion = round(1.0 / (1.0 + std / 20), 4)
            else:
                ua.cohesion = 0.5

            unit.cohesion = ua.cohesion

            # Yedeklilik: birimde kaç oyuncu var / minimum gerekli
            min_needed = max(2, len(unit.player_indices) - 1)
            active = sum(
                1 for p in unit.player_indices if p not in missing
            )
            ua.redundancy = round(
                max(0, (active - min_needed)) / max(len(unit.player_indices), 1), 4,
            )

            total_failure += ua.failure_prob * unit.weight
            report.unit_analyses.append(ua)

            # Uyarı sinyalleri
            if ua.failure_prob > self.FAILURE_THRESHOLD:
                if unit.unit_type == "defense":
                    report.defense_alert = True
                elif unit.unit_type == "midfield":
                    report.midfield_alert = True
                elif unit.unit_type == "attack":
                    report.attack_alert = True

        # Global metrikler
        total_weight = sum(u.weight for u in units) or 1.0
        report.vulnerability_index = round(total_failure / total_weight, 4)
        report.team_cohesion = round(
            np.mean([ua.cohesion for ua in report.unit_analyses]) if report.unit_analyses else 0,
            4,
        )

        report.recommendation = self._advice(report)
        return report

    def missing_player_impact(self, team: str,
                                units: list[TacticalUnit],
                                player_ratings: np.ndarray,
                                player_idx: int) -> dict:
        """Tek bir oyuncunun eksiklik etkisini hesapla."""
        # Tam kadro
        full = self.analyze_team(team, units, player_ratings)
        # Eksik kadro
        missing = self.analyze_team(
            team, units, player_ratings, missing_players=[player_idx],
        )

        vuln_delta = missing.vulnerability_index - full.vulnerability_index

        return {
            "player_idx": player_idx,
            "vulnerability_before": full.vulnerability_index,
            "vulnerability_after": missing.vulnerability_index,
            "vulnerability_delta": round(vuln_delta, 4),
            "defense_alert_changed": missing.defense_alert and not full.defense_alert,
            "critical": vuln_delta > 0.15,
        }

    def _advice(self, report: HypergraphReport) -> str:
        alerts = []
        if report.defense_alert:
            alerts.append("SAVUNMA ÇÖKÜyor")
        if report.midfield_alert:
            alerts.append("ORTA SAHA zayıf")
        if report.attack_alert:
            alerts.append("HÜCUM bloğu kırılgan")

        if alerts:
            return (
                f"UYARI: {', '.join(alerts)}! "
                f"Kırılganlık={report.vulnerability_index:.2f}, "
                f"Uyum={report.team_cohesion:.2f}. "
                f"Bahis güvenini düşür."
            )
        if report.team_cohesion < self.COHESION_THRESHOLD:
            return (
                f"Düşük takım uyumu: {report.team_cohesion:.2f}. "
                f"Bireysel yetenekler iyi olsa bile takım oyunu zayıf."
            )
        return (
            f"Takım yapısı sağlam: uyum={report.team_cohesion:.2f}, "
            f"kırılganlık={report.vulnerability_index:.2f}."
        )
