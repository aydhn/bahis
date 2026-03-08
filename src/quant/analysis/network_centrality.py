"""
network_centrality.py – Ağ Bilimi ve Pas Bağlantıları (Network Science).

Takımın en golcü oyuncusu sakatlanınca oranlar değişir.
Ama "pas trafiğini yöneten" gizli kahraman (Regista) sakatlanınca?
Bunu sadece Ağ Bilimi çözer.

Metrikler:
  PageRank      → Oyunu kim yönlendiriyor? (Google'ın algoritması)
  Betweenness   → Kilit pas istasyonu kim? (Köprü oyuncu)
  Degree        → En çok pas alan/veren kim?
  Closeness     → Kim herkese en kolay ulaşır?
  Eigenvector   → Önemli oyuncularla pas yapan kim önemlidir?

Sinyal:
  PageRank'i en yüksek oyuncu maçta yoksa →
  Takımın xG'sini %20 düşüren "Penalty Factor" katsayısı üretilir.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from loguru import logger

try:
    import networkx as nx
    NX_OK = True
except ImportError:
    NX_OK = False
    logger.warning("networkx yüklü değil – centrality hesaplanamaz.")


@dataclass
class PlayerCentrality:
    """Bir oyuncunun ağ metrikleri."""
    player_id: str
    name: str
    team: str
    position: str = ""
    pagerank: float = 0.0
    betweenness: float = 0.0
    degree: float = 0.0
    closeness: float = 0.0
    eigenvector: float = 0.0
    composite_score: float = 0.0
    is_key_player: bool = False
    rank_in_team: int = 0


@dataclass
class AbsencePenalty:
    """Eksik oyuncunun takıma etkisi."""
    player: str
    team: str
    penalty_factor: float = 1.0   # 1.0 = etki yok, 0.80 = -%20 xG
    xg_adjustment: float = 0.0    # Mutlak xG düzeltmesi
    reason: str = ""
    severity: str = "low"         # low / medium / high / critical


class PassNetworkAnalyzer:
    """Pas ağı analizi ile oyuncu önem sıralaması.

    Kullanım:
        analyzer = PassNetworkAnalyzer()
        # Pas verisi yükle
        analyzer.load_passes(team="Galatasaray", passes=pass_data)
        # Centrality hesapla
        rankings = analyzer.calculate_centrality("Galatasaray")
        # Eksik oyuncu etkisi
        penalty = analyzer.absence_impact("Galatasaray", missing=["Icardi"])
    """

    # Penalty ağırlıkları (PageRank ne kadar yüksekse, eksiklik o kadar etkili)
    PENALTY_WEIGHTS = {
        "pagerank": 0.35,
        "betweenness": 0.30,
        "eigenvector": 0.20,
        "degree": 0.15,
    }

    # Pozisyon bazlı ek çarpan
    POSITION_MULTIPLIER = {
        "GK": 0.3,
        "DEF": 0.7,
        "MID": 1.0,    # Orta saha eksikliği en kritik
        "FWD": 0.9,
    }

    def __init__(self):
        self._graphs: dict[str, Any] = {}  # team → DiGraph
        self._centrality_cache: dict[str, list[PlayerCentrality]] = {}
        self._pass_counts: dict[str, int] = {}
        logger.debug("PassNetworkAnalyzer başlatıldı.")

    # ═══════════════════════════════════════════
    #  PAS VERİSİ YÜKLEME
    # ═══════════════════════════════════════════
    def load_passes(self, team: str, passes: list[dict]):
        """Pas verisini ağa yükle.

        passes: [
            {"from": "player_a", "to": "player_b", "count": 45, "success_rate": 0.85},
            ...
        ]
        """
        if not NX_OK:
            return

        G = nx.DiGraph()
        total_passes = 0

        for p in passes:
            src = p.get("from", "")
            dst = p.get("to", "")
            count = p.get("count", 1)
            success = p.get("success_rate", 0.8)

            if src and dst:
                weight = count * success  # Ağırlık = pas sayısı × başarı oranı
                if G.has_edge(src, dst):
                    G[src][dst]["weight"] += weight
                    G[src][dst]["count"] += count
                else:
                    G.add_edge(src, dst, weight=weight, count=count)
                total_passes += count

        # Oyuncu meta verisi ekle
        for p in passes:
            for key in ("from", "to"):
                player = p.get(key, "")
                if player and player in G.nodes:
                    G.nodes[player].update({
                        "position": p.get(f"{key}_position", ""),
                        "team": team,
                    })

        self._graphs[team] = G
        self._pass_counts[team] = total_passes
        self._centrality_cache.pop(team, None)  # Cache temizle

        logger.info(
            f"[Network] {team}: {G.number_of_nodes()} oyuncu, "
            f"{G.number_of_edges()} bağlantı, {total_passes} pas"
        )

    def load_from_match_data(self, team: str, match_data: dict):
        """Maç verisinden basit pas ağı oluştur (detaylı veri yoksa)."""
        if not NX_OK:
            return

        lineup = match_data.get("lineup", [])
        if len(lineup) < 8:
            return

        G = nx.DiGraph()

        # Basit model: pozisyon bazlı pas bağlantıları
        positions = {"GK": [], "DEF": [], "MID": [], "FWD": []}
        for p in lineup:
            pos = p.get("position", "MID")[:3].upper()
            if pos not in positions:
                pos = "MID"
            positions[pos].append(p.get("name", ""))
            G.add_node(p.get("name", ""), position=pos, team=team)

        # GK → DEF → MID → FWD zincirleme bağlantılar
        chain = ["GK", "DEF", "MID", "FWD"]
        for i in range(len(chain) - 1):
            for src in positions[chain[i]]:
                for dst in positions[chain[i + 1]]:
                    G.add_edge(src, dst, weight=10, count=10)

        # Aynı bölge içi bağlantılar
        for pos, players in positions.items():
            for i, p1 in enumerate(players):
                for p2 in players[i + 1:]:
                    G.add_edge(p1, p2, weight=15, count=15)
                    G.add_edge(p2, p1, weight=15, count=15)

        self._graphs[team] = G
        self._centrality_cache.pop(team, None)

    # ═══════════════════════════════════════════
    #  CENTRALITY HESAPLAMA
    # ═══════════════════════════════════════════
    def calculate_centrality(self, team: str) -> list[PlayerCentrality]:
        """Tüm centrality metriklerini hesapla ve sırala."""
        if team in self._centrality_cache:
            return self._centrality_cache[team]

        G = self._graphs.get(team)
        if not G or not NX_OK:
            return []

        # PageRank
        try:
            pr = nx.pagerank(G, weight="weight", alpha=0.85)
        except Exception:
            pr = {n: 1.0 / G.number_of_nodes() for n in G.nodes}

        # Betweenness Centrality
        try:
            bc = nx.betweenness_centrality(G, weight="weight", normalized=True)
        except Exception:
            bc = {n: 0.0 for n in G.nodes}

        # Degree Centrality (ağırlıklı)
        try:
            dc = nx.degree_centrality(G)
        except Exception:
            dc = {n: 0.0 for n in G.nodes}

        # Closeness Centrality
        try:
            cc = nx.closeness_centrality(G)
        except Exception:
            cc = {n: 0.0 for n in G.nodes}

        # Eigenvector Centrality
        try:
            ec = nx.eigenvector_centrality(G, weight="weight", max_iter=500)
        except Exception:
            ec = {n: 0.0 for n in G.nodes}

        # Composite score
        results = []
        for node in G.nodes:
            composite = (
                pr.get(node, 0) * self.PENALTY_WEIGHTS["pagerank"] +
                bc.get(node, 0) * self.PENALTY_WEIGHTS["betweenness"] +
                ec.get(node, 0) * self.PENALTY_WEIGHTS["eigenvector"] +
                dc.get(node, 0) * self.PENALTY_WEIGHTS["degree"]
            )

            results.append(PlayerCentrality(
                player_id=node,
                name=node,
                team=team,
                position=G.nodes[node].get("position", ""),
                pagerank=float(pr.get(node, 0)),
                betweenness=float(bc.get(node, 0)),
                degree=float(dc.get(node, 0)),
                closeness=float(cc.get(node, 0)),
                eigenvector=float(ec.get(node, 0)),
                composite_score=float(composite),
            ))

        # Sırala ve rank ata
        results.sort(key=lambda x: x.composite_score, reverse=True)
        for i, r in enumerate(results):
            r.rank_in_team = i + 1
            r.is_key_player = i < 3  # İlk 3 = kilit oyuncu

        self._centrality_cache[team] = results
        return results

    # ═══════════════════════════════════════════
    #  EKSİK OYUNCU ETKİSİ
    # ═══════════════════════════════════════════
    def absence_impact(self, team: str,
                        missing: list[str]) -> list[AbsencePenalty]:
        """Eksik oyuncuların takım performansına etkisi.

        Returns:
            Her eksik oyuncu için bir AbsencePenalty.
            penalty_factor: 1.0 = etki yok, 0.80 = -%20 xG düzeltmesi
        """
        rankings = self.calculate_centrality(team)
        if not rankings:
            return []

        max_composite = max(r.composite_score for r in rankings) or 1.0

        penalties = []
        for player_name in missing:
            player = next(
                (r for r in rankings if r.name.lower() == player_name.lower()),
                None,
            )
            if not player:
                penalties.append(AbsencePenalty(
                    player=player_name, team=team,
                    penalty_factor=0.95,
                    reason="Oyuncu ağda bulunamadı – varsayılan ceza.",
                    severity="low",
                ))
                continue

            # Etki oranı: oyuncunun önem skoru / maksimum önem
            importance = player.composite_score / max_composite

            # Pozisyon çarpanı
            pos = player.position[:3].upper() if player.position else "MID"
            pos_mult = self.POSITION_MULTIPLIER.get(pos, 0.8)

            # Penalty hesapla: 0.0 (yok sayılamaz) → 0.30 (kritik)
            raw_penalty = importance * pos_mult * 0.30
            penalty_factor = max(1.0 - raw_penalty, 0.65)  # Minimum %65

            # Şiddet sınıflandırma
            if raw_penalty >= 0.20:
                severity = "critical"
            elif raw_penalty >= 0.12:
                severity = "high"
            elif raw_penalty >= 0.06:
                severity = "medium"
            else:
                severity = "low"

            reason = (
                f"PageRank #{player.rank_in_team}, "
                f"Betweenness={player.betweenness:.3f}, "
                f"Pozisyon={pos}"
            )

            penalties.append(AbsencePenalty(
                player=player_name,
                team=team,
                penalty_factor=round(penalty_factor, 3),
                xg_adjustment=round(raw_penalty, 3),
                reason=reason,
                severity=severity,
            ))

            logger.info(
                f"[Network] {team} – {player_name} EKSİK: "
                f"penalty={penalty_factor:.0%}, severity={severity}"
            )

        return penalties

    def combined_penalty(self, team: str,
                          missing: list[str]) -> float:
        """Tüm eksik oyuncuların toplam penalty faktörü.

        Returns: 0.0-1.0 (xG çarpanı). Örn: 0.75 = xG %25 düşürüldü.
        """
        penalties = self.absence_impact(team, missing)
        if not penalties:
            return 1.0

        factor = 1.0
        for p in penalties:
            factor *= p.penalty_factor

        return max(factor, 0.50)  # Minimum %50'ye düşebilir

    # ═══════════════════════════════════════════
    #  TAKIM KARŞILAŞTIRMA
    # ═══════════════════════════════════════════
    def compare_teams(self, team_a: str, team_b: str) -> dict:
        """İki takımın ağ yapısını karşılaştır."""
        rank_a = self.calculate_centrality(team_a)
        rank_b = self.calculate_centrality(team_b)

        def _team_summary(rankings):
            if not rankings:
                return {"n_players": 0}
            scores = [r.composite_score for r in rankings]
            return {
                "n_players": len(rankings),
                "top_player": rankings[0].name if rankings else "",
                "avg_composite": float(np.mean(scores)),
                "max_composite": float(max(scores)),
                "concentration": float(max(scores) / (np.mean(scores) or 1)),
            }

        a_summary = _team_summary(rank_a)
        b_summary = _team_summary(rank_b)

        return {
            "team_a": {**a_summary, "team": team_a},
            "team_b": {**b_summary, "team": team_b},
            "network_advantage": (
                "A" if a_summary.get("avg_composite", 0) > b_summary.get("avg_composite", 0)
                else "B"
            ),
            "a_more_dependent": a_summary.get("concentration", 1) > 2.0,
            "b_more_dependent": b_summary.get("concentration", 1) > 2.0,
        }

    @property
    def loaded_teams(self) -> list[str]:
        return list(self._graphs.keys())
