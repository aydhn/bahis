"""
homology_scanner.py – Persistent Homology (Topolojik Veri Analizi).

Veriyi sadece sayı olarak değil, bir "Şekil" olarak görür.
Veri kümesindeki "Delikleri" ve "Döngüleri" bulur.

Kavramlar:
  - Point Cloud: Oyuncuların saha koordinatları
  - Simplicial Complex: Noktalar arası bağlantılar (Rips/Vietoris-Rips)
  - Betti Numbers: Topolojik özellik sayıları
    β₀: Bağlı bileşen sayısı (izole gruplar)
    β₁: 1-döngü sayısı (delikler/halka yapıları)
    β₂: 2-döngü sayısı (boşluklar)
  - Persistence Diagram: (birth, death) çiftleri
  - Persistence: death - birth (uzun ömürlü = yapısal, kısa = gürültü)
  - Barcode: Topolojik özelliklerin ömür çubukları

Sinyaller:
  - Kısa ömürlü döngüler çok → Takım paniklemiş (kaotik hareket)
  - Uzun ömürlü döngüler → Takımın bir planı var (organize)
  - β₀ yüksek → Takım kopuk gruplar halinde (bireysel oyun)
  - β₁ yüksek → Pas döngüleri mevcut (akıcı oyun)

Teknoloji: ripser veya gudhi
Fallback: Manuel Rips complex + basit homoloji (scipy)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from loguru import logger

try:
    from ripser import ripser
    RIPSER_OK = True
except ImportError:
    RIPSER_OK = False
    logger.debug("ripser yüklü değil – manuel homoloji fallback.")

try:
    from scipy.spatial.distance import pdist, squareform
    SCIPY_DIST_OK = True
except ImportError:
    SCIPY_DIST_OK = False


# ═══════════════════════════════════════════════
#  VERİ YAPILARI
# ═══════════════════════════════════════════════
@dataclass
class PersistencePair:
    """Kalıcılık çifti (birth, death)."""
    dimension: int = 0       # 0: bileşen, 1: döngü, 2: boşluk
    birth: float = 0.0       # Doğum (oluşum mesafesi)
    death: float = 0.0       # Ölüm (yok olma mesafesi)
    persistence: float = 0.0  # death - birth


@dataclass
class HomologyReport:
    """Topolojik analiz raporu."""
    team: str = ""
    match_id: str = ""
    # Betti sayıları
    betti_0: int = 0             # Bağlı bileşen sayısı
    betti_1: int = 0             # 1-döngü sayısı
    betti_2: int = 0             # 2-boşluk sayısı
    # Kalıcılık istatistikleri
    n_features: int = 0          # Toplam topolojik özellik
    avg_persistence: float = 0.0  # Ortalama kalıcılık
    max_persistence: float = 0.0  # Maksimum kalıcılık
    noise_ratio: float = 0.0     # Gürültü oranı (kısa ömürlü / toplam)
    # Skor
    organization_score: float = 0.0   # 0-1 organizasyon skoru
    compactness_score: float = 0.0    # 0-1 sıkılık skoru
    connectivity_score: float = 0.0   # 0-1 bağlantısallık skoru
    # Sinyaller
    team_panicking: bool = False      # Takım paniklemiş
    organized_play: bool = False      # Organize oyun
    isolated_groups: bool = False     # Kopuk gruplar
    passing_cycles: bool = False      # Pas döngüleri mevcut
    formation_broken: bool = False    # Formasyon bozulmuş
    # Meta
    n_players: int = 0
    method: str = ""
    recommendation: str = ""
    # Ham veriler
    persistence_pairs: list[PersistencePair] = field(default_factory=list)


# ═══════════════════════════════════════════════
#  MANUEL RIPS COMPLEX & HOMOLOJI
# ═══════════════════════════════════════════════
def compute_distance_matrix(points: np.ndarray) -> np.ndarray:
    """Mesafe matrisini hesapla."""
    if SCIPY_DIST_OK:
        return squareform(pdist(points))
    n = len(points)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt(np.sum((points[i] - points[j]) ** 2))
            D[i, j] = d
            D[j, i] = d
    return D


def manual_persistence(points: np.ndarray,
                          max_dim: int = 1,
                          max_edge: float = 30.0) -> list[PersistencePair]:
    """Manuel kalıcı homoloji (Union-Find ile H₀).

    Tam Rips complex hesaplama ağır olduğu için,
    sadece H₀ (bağlı bileşenler) ve basit H₁ tahmini yapar.
    """
    n = len(points)
    if n < 2:
        return []

    D = compute_distance_matrix(points)
    pairs: list[PersistencePair] = []

    # ── H₀: Bağlı bileşenler (Kruskal/Union-Find) ──
    parent = list(range(n))
    rank = [0] * n

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> bool:
        rx, ry = find(x), find(y)
        if rx == ry:
            return False
        if rank[rx] < rank[ry]:
            rx, ry = ry, rx
        parent[ry] = rx
        if rank[rx] == rank[ry]:
            rank[rx] += 1
        return True

    # Kenarları mesafeye göre sırala
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if D[i, j] <= max_edge:
                edges.append((D[i, j], i, j))
    edges.sort()

    # Her bileşenin doğum zamanı
    births = {i: 0.0 for i in range(n)}
    components = n

    for dist, u, v in edges:
        ru, rv = find(u), find(v)
        if ru != rv:
            # Bir bileşen ölüyor (birleşiyor)
            dying = rv if rank[ru] >= rank[rv] else ru
            pairs.append(PersistencePair(
                dimension=0,
                birth=births.get(dying, 0.0),
                death=dist,
                persistence=dist - births.get(dying, 0.0),
            ))
            union(u, v)
            components -= 1

    # Hayatta kalan bileşenler (sonsuz ömür)
    alive = set()
    for i in range(n):
        alive.add(find(i))
    for comp in alive:
        pairs.append(PersistencePair(
            dimension=0,
            birth=0.0,
            death=float("inf"),
            persistence=float("inf"),
        ))

    # ── H₁: Basit döngü tahmini ──
    if max_dim >= 1 and len(edges) > n:
        # Minimum spanning tree kenar sayısı = n-1
        # Fazla kenarlar döngü oluşturur
        mst_edge_count = n - 1
        non_tree_edges = edges[mst_edge_count:]

        for dist, u, v in non_tree_edges[:20]:  # İlk 20 döngü
            pairs.append(PersistencePair(
                dimension=1,
                birth=dist * 0.8,
                death=dist,
                persistence=dist * 0.2,
            ))

    return pairs


# ═══════════════════════════════════════════════
#  HOMOLOGY SCANNER (Ana Sınıf)
# ═══════════════════════════════════════════════
class HomologyScanner:
    """Topolojik veri analizi ile takım organizasyonu tespiti.

    Kullanım:
        hs = HomologyScanner()

        # Oyuncu pozisyonları (x, y) – saha koordinatları
        positions = np.array([
            [15, 34], [25, 20], [25, 48],   # Defans
            [45, 15], [45, 34], [45, 53],   # Orta saha
            [65, 25], [65, 43],             # Kanat
            [78, 34],                        # Forvet
            [5, 34],                         # Kaleci
        ])

        report = hs.analyze_team(
            positions, team="Galatasaray",
            match_id="gs_fb",
        )

        if report.team_panicking:
            increase_over_bet()
    """

    NOISE_THRESHOLD = 2.0            # Kalıcılık < bu → gürültü
    PANIC_NOISE_RATIO = 0.7          # Gürültü oranı > bu → panik
    ORGANIZED_PERSISTENCE = 10.0     # Uzun ömürlü eşik
    ISOLATED_BETTI0 = 3              # β₀ > bu → kopuk gruplar

    def __init__(self, max_dim: int = 1, max_edge: float = 50.0):
        self._max_dim = max_dim
        self._max_edge = max_edge
        logger.debug(
            f"[Homology] Scanner başlatıldı: max_dim={max_dim}, "
            f"max_edge={max_edge}"
        )

    def analyze_team(self, positions: np.ndarray,
                       team: str = "",
                       match_id: str = "",
                       velocities: np.ndarray | None = None) -> HomologyReport:
        """Takım pozisyonlarından topolojik analiz."""
        report = HomologyReport(team=team, match_id=match_id)
        points = np.array(positions, dtype=np.float64)

        if len(points) < 3:
            report.recommendation = "Yetersiz oyuncu verisi (min 3)."
            report.method = "none"
            return report

        report.n_players = len(points)

        # Hız vektörleri varsa pozisyonlara ekle (4D gömme)
        if velocities is not None and len(velocities) == len(points):
            vel = np.array(velocities, dtype=np.float64)
            augmented = np.hstack([points, vel * 5])  # Hız ağırlıklı
        else:
            augmented = points

        # Kalıcı homoloji hesapla
        if RIPSER_OK:
            try:
                result = ripser(
                    augmented, maxdim=self._max_dim,
                    thresh=self._max_edge,
                )
                pairs = self._parse_ripser(result)
                report.method = "ripser"
            except Exception:
                pairs = manual_persistence(
                    augmented, self._max_dim, self._max_edge,
                )
                report.method = "manual_rips"
        else:
            pairs = manual_persistence(
                augmented, self._max_dim, self._max_edge,
            )
            report.method = "manual_rips"

        report.persistence_pairs = pairs
        report.n_features = len(pairs)

        # Betti sayıları (sonsuz ömürlüler)
        finite_pairs = [p for p in pairs if p.persistence < float("inf")]
        inf_pairs = [p for p in pairs if p.persistence == float("inf")]

        report.betti_0 = sum(
            1 for p in inf_pairs if p.dimension == 0
        )
        report.betti_1 = sum(
            1 for p in pairs if p.dimension == 1
        )

        # Kalıcılık istatistikleri (sonlu olanlar)
        if finite_pairs:
            persistences = [p.persistence for p in finite_pairs]
            report.avg_persistence = round(float(np.mean(persistences)), 3)
            report.max_persistence = round(float(max(persistences)), 3)

            # Gürültü oranı
            noise_count = sum(
                1 for p in persistences if p < self.NOISE_THRESHOLD
            )
            report.noise_ratio = round(noise_count / len(persistences), 3)

        # Skorlar
        report.organization_score = self._org_score(report, finite_pairs)
        report.compactness_score = self._compactness(points)
        report.connectivity_score = self._connectivity(report)

        # Sinyaller
        if report.noise_ratio > self.PANIC_NOISE_RATIO:
            report.team_panicking = True

        long_lived = [
            p for p in finite_pairs
            if p.persistence > self.ORGANIZED_PERSISTENCE
        ]
        if len(long_lived) >= 2:
            report.organized_play = True

        if report.betti_0 > self.ISOLATED_BETTI0:
            report.isolated_groups = True

        if report.betti_1 >= 2:
            report.passing_cycles = True

        if report.organization_score < 0.3:
            report.formation_broken = True

        report.recommendation = self._advice(report)
        return report

    def compare_teams(self, home_pos: np.ndarray,
                        away_pos: np.ndarray,
                        match_id: str = "") -> dict:
        """İki takımın topolojik karşılaştırması."""
        home_report = self.analyze_team(home_pos, "home", match_id)
        away_report = self.analyze_team(away_pos, "away", match_id)

        return {
            "home_org": home_report.organization_score,
            "away_org": away_report.organization_score,
            "org_advantage": round(
                home_report.organization_score - away_report.organization_score,
                3,
            ),
            "home_panicking": home_report.team_panicking,
            "away_panicking": away_report.team_panicking,
            "home_compact": home_report.compactness_score,
            "away_compact": away_report.compactness_score,
            "home_betti": (home_report.betti_0, home_report.betti_1),
            "away_betti": (away_report.betti_0, away_report.betti_1),
        }

    def _parse_ripser(self, result: dict) -> list[PersistencePair]:
        """Ripser sonucunu PersistencePair listesine çevir."""
        pairs = []
        diagrams = result.get("dgms", [])
        for dim, dgm in enumerate(diagrams):
            for birth, death in dgm:
                pers = float("inf") if np.isinf(death) else death - birth
                pairs.append(PersistencePair(
                    dimension=dim,
                    birth=float(birth),
                    death=float(death),
                    persistence=float(pers),
                ))
        return pairs

    def _org_score(self, report: HomologyReport,
                     finite_pairs: list[PersistencePair]) -> float:
        """Organizasyon skoru (0-1)."""
        if not finite_pairs:
            return 0.5

        # Uzun ömürlü / toplam oranı
        long_ratio = sum(
            1 for p in finite_pairs
            if p.persistence > self.NOISE_THRESHOLD
        ) / max(len(finite_pairs), 1)

        # Gürültü cezası
        noise_penalty = report.noise_ratio * 0.5

        # β₀ cezası (fazla kopukluk = kötü)
        betti_penalty = max(0, (report.betti_0 - 2) * 0.1)

        score = long_ratio - noise_penalty - betti_penalty
        return round(float(np.clip(score, 0, 1)), 3)

    def _compactness(self, points: np.ndarray) -> float:
        """Sıkılık skoru (0-1). Takım ne kadar kompakt?"""
        if len(points) < 2:
            return 0.5
        centroid = np.mean(points, axis=0)
        distances = np.sqrt(np.sum((points - centroid) ** 2, axis=1))
        avg_dist = np.mean(distances)
        # Normalize (saha yarısı ~52.5m baz)
        score = 1 - min(avg_dist / 30, 1.0)
        return round(float(np.clip(score, 0, 1)), 3)

    def _connectivity(self, report: HomologyReport) -> float:
        """Bağlantısallık skoru (0-1)."""
        if report.n_players == 0:
            return 0.0
        # İdeal: β₀=1 (tamamen bağlı), β₁>0 (döngüler)
        betti0_score = 1.0 / max(report.betti_0, 1)
        betti1_bonus = min(report.betti_1 * 0.2, 0.3)
        return round(float(np.clip(betti0_score + betti1_bonus, 0, 1)), 3)

    def _advice(self, r: HomologyReport) -> str:
        if r.team_panicking:
            return (
                f"TAKlM PANİKLEMİŞ: Gürültü oranı={r.noise_ratio:.0%}, "
                f"β₀={r.betti_0} kopuk grup. "
                f"Organizasyon={r.organization_score:.0%}. "
                f"Rakip takıma ÜST SİNYALİ!"
            )
        if r.formation_broken:
            return (
                f"FORMASYON BOZUK: Org={r.organization_score:.0%}, "
                f"kompaktlık={r.compactness_score:.0%}. "
                f"Savunma zafiyeti var."
            )
        if r.organized_play:
            return (
                f"ORGANİZE OYUN: Uzun ömürlü yapılar tespit, "
                f"β₁={r.betti_1} pas döngüsü, "
                f"org={r.organization_score:.0%}. "
                f"Takımın bir planı var."
            )
        return (
            f"Normal: β₀={r.betti_0}, β₁={r.betti_1}, "
            f"org={r.organization_score:.0%}, "
            f"kompakt={r.compactness_score:.0%}."
        )
