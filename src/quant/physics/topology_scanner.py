"""
topology_scanner.py – Topological Data Analysis (TDA) ile Formasyon Analizi.

Futbol bir geometridir. Takımların sahadaki dizilişi bir "Şekil"
oluşturur. Bu şeklin bozulması gol habercisidir.

Süreç:
  1. Oyuncu koordinatlarını "Nokta Bulutu" (Point Cloud) olarak al
  2. Persistent Homology ile topolojik yapıyı analiz et
  3. Delikler (H1) ve bağlı bileşenler (H0) takip et
  4. Formasyon bütünlüğü bozulursa → "Savunma Çöküyor" uyarısı

Topolojik Kavramlar:
  - H0 (Bağlı Bileşenler): Oyuncuların kaç gruba ayrıldığı
    → Normal: 1 ana küme. Panik: 3-4 dağınık küme
  - H1 (Delikler/Döngüler): Savunma hattındaki boşluklar
    → Yüksek persistence = büyük açık
  - Persistence Diagram: Topolojik özelliklerin doğum/ölüm çizelgesi
  - Wasserstein Distance: İki formasyon arasındaki topolojik mesafe

Teknoloji: giotto-tda veya ripser
Fallback: scipy + convex hull + Voronoi analizi
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from loguru import logger

try:
    from gtda.homology import VietorisRipsPersistence
    from gtda.diagrams import PersistenceEntropy, Amplitude
    GTDA_OK = True
except ImportError:
    GTDA_OK = False
    logger.info("giotto-tda yüklü değil – geometrik fallback aktif.")

try:
    from scipy.spatial import ConvexHull, Voronoi, Delaunay
    from scipy.spatial.distance import pdist, squareform
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False


# ═══════════════════════════════════════════════
#  VERİ MODELLERİ
# ═══════════════════════════════════════════════
@dataclass
class FormationState:
    """Takım formasyon durumu."""
    team: str = ""
    n_players: int = 0
    # Geometrik metrikler
    convex_hull_area: float = 0.0       # Kaplama alanı (m^2)
    compactness: float = 0.0            # Sıkılık (0-1, 1=çok sıkı)
    stretch_index: float = 0.0          # Genişleme (0-1, 1=çok açık)
    center_of_mass: tuple = (0, 0)      # Ağırlık merkezi
    defensive_line_height: float = 0.0  # Savunma hattı yüksekliği (m)
    width: float = 0.0                  # Takım genişliği (m)
    depth: float = 0.0                  # Takım derinliği (m)
    # Topolojik metrikler
    h0_components: int = 1              # Bağlı bileşen sayısı
    h1_holes: int = 0                   # Delik/boşluk sayısı
    persistence_entropy: float = 0.0    # Topolojik karmaşıklık
    max_persistence: float = 0.0        # En uzun yaşayan topolojik özellik
    formation_integrity: float = 1.0    # Formasyon bütünlüğü (0-1)
    method: str = "geometric"


@dataclass
class FormationAlert:
    """Formasyon bozulma uyarısı."""
    team: str = ""
    match_id: str = ""
    alert_type: str = ""          # collapse | gap | stretch | panic
    severity: str = "medium"      # low | medium | high | critical
    integrity: float = 0.0
    description: str = ""
    details: dict = field(default_factory=dict)


@dataclass
class FormationComparison:
    """İki formasyon durumunun karşılaştırması."""
    team: str = ""
    state_before: FormationState = field(default_factory=FormationState)
    state_after: FormationState = field(default_factory=FormationState)
    integrity_change: float = 0.0
    area_change_pct: float = 0.0
    compactness_change: float = 0.0
    is_deteriorating: bool = False
    interpretation: str = ""


class TopologyScanner:
    """Topolojik Veri Analizi ile formasyon izleyici.

    Kullanım:
        scanner = TopologyScanner()
        # Oyuncu koordinatlarından formasyon analizi
        state = scanner.analyze_formation(coords, team="home")
        # Formasyon bozulma kontrolü
        alerts = scanner.check_integrity(coords, team="home")
        # İki durum karşılaştır
        comparison = scanner.compare(state_before, state_after)
    """

    # Eşik değerleri (saha: 105x68 metre)
    COMPACT_THRESHOLD = 800       # m^2 altı = çok sıkı
    STRETCHED_THRESHOLD = 2000    # m^2 üstü = çok açık
    INTEGRITY_ALERT = 0.50        # %50 altı = tehlikeli
    PANIC_THRESHOLD = 0.30        # %30 altı = panik
    MIN_PLAYERS = 5               # Minimum analiz edilebilir oyuncu

    def __init__(self, homology_dims: list[int] | None = None):
        """
        Args:
            homology_dims: Hesaplanacak homoloji boyutları [0, 1]
        """
        self._dims = homology_dims or [0, 1]
        self._history: dict[str, list[FormationState]] = {}
        self._persistence_model = None

        if GTDA_OK:
            try:
                self._persistence_model = VietorisRipsPersistence(
                    homology_dimensions=self._dims,
                    n_jobs=-1,
                )
            except Exception as e:
                logger.debug(f"[TDA] giotto-tda init hatası: {e}")

        logger.debug(
            f"[TDA] TopologyScanner başlatıldı "
            f"(method={'giotto-tda' if GTDA_OK else 'geometric'})"
        )

    # ═══════════════════════════════════════════
    #  FORMASYON ANALİZİ
    # ═══════════════════════════════════════════
    def analyze_formation(self, coords: list[tuple[float, float]],
                           team: str = "") -> FormationState:
        """Oyuncu koordinatlarından formasyon durumunu analiz et.

        Args:
            coords: [(x1, y1), (x2, y2), ...] saha koordinatları (metre)
            team: Takım adı
        """
        state = FormationState(team=team, n_players=len(coords))

        if len(coords) < self.MIN_PLAYERS:
            return state

        points = np.array(coords, dtype=np.float64)

        # Geometrik metrikler
        state = self._geometric_analysis(points, state)

        # Topolojik metrikler
        if GTDA_OK and self._persistence_model:
            state = self._tda_analysis(points, state)
            state.method = "tda"
        else:
            state = self._fallback_topology(points, state)
            state.method = "geometric"

        # Formasyon bütünlüğü hesapla
        state.formation_integrity = self._calculate_integrity(state)

        # Geçmişe ekle
        if team:
            if team not in self._history:
                self._history[team] = []
            self._history[team].append(state)
            if len(self._history[team]) > 200:
                self._history[team] = self._history[team][-100:]

        return state

    def _geometric_analysis(self, points: np.ndarray,
                              state: FormationState) -> FormationState:
        """Geometrik metrikler hesapla."""
        # Ağırlık merkezi
        com = points.mean(axis=0)
        state.center_of_mass = (round(float(com[0]), 1), round(float(com[1]), 1))

        # Genişlik ve derinlik
        state.width = float(points[:, 1].max() - points[:, 1].min())
        state.depth = float(points[:, 0].max() - points[:, 0].min())

        # Convex Hull alanı
        if SCIPY_OK and len(points) >= 3:
            try:
                hull = ConvexHull(points)
                state.convex_hull_area = round(float(hull.volume), 1)  # 2D'de volume = alan
            except Exception:
                state.convex_hull_area = state.width * state.depth * 0.5

        # Sıkılık: oyuncular arası ortalama mesafe
        if len(points) > 1:
            distances = pdist(points) if SCIPY_OK else self._manual_pdist(points)
            avg_dist = float(np.mean(distances))
            max_possible = np.sqrt(105 ** 2 + 68 ** 2) / 2
            state.compactness = round(1.0 - min(avg_dist / max_possible, 1.0), 3)

            # Stretch index
            state.stretch_index = round(float(np.std(distances)) / max(avg_dist, 1), 3)

        # Savunma hattı yüksekliği (en geri 4 oyuncu)
        sorted_x = np.sort(points[:, 0])
        if len(sorted_x) >= 4:
            state.defensive_line_height = round(float(np.mean(sorted_x[:4])), 1)

        return state

    def _tda_analysis(self, points: np.ndarray,
                        state: FormationState) -> FormationState:
        """giotto-tda ile topolojik analiz."""
        try:
            # Persistent Homology
            pts_3d = points.reshape(1, -1, 2)  # (1, n_points, 2)
            diagrams = self._persistence_model.fit_transform(pts_3d)

            # H0: Bağlı bileşenler
            h0_mask = diagrams[0][:, 2] == 0
            h0_diag = diagrams[0][h0_mask]
            # Uzun yaşayan bileşenler = farklı kümeler
            h0_lifetimes = h0_diag[:, 1] - h0_diag[:, 0]
            significant_h0 = np.sum(h0_lifetimes > np.median(h0_lifetimes) * 1.5)
            state.h0_components = max(1, int(significant_h0) + 1)

            # H1: Delikler (savunma boşlukları)
            h1_mask = diagrams[0][:, 2] == 1
            h1_diag = diagrams[0][h1_mask]
            if len(h1_diag) > 0:
                h1_lifetimes = h1_diag[:, 1] - h1_diag[:, 0]
                state.h1_holes = int(np.sum(h1_lifetimes > np.percentile(h1_lifetimes, 75)))
                state.max_persistence = round(float(np.max(h1_lifetimes)), 3)

            # Persistence Entropy
            try:
                pe = PersistenceEntropy()
                entropy = pe.fit_transform(diagrams)
                state.persistence_entropy = round(float(entropy[0].mean()), 4)
            except Exception:
                pass

        except Exception as e:
            logger.debug(f"[TDA] Analiz hatası: {e}")

        return state

    def _fallback_topology(self, points: np.ndarray,
                             state: FormationState) -> FormationState:
        """giotto-tda yoksa geometrik yaklaşım."""
        if not SCIPY_OK or len(points) < 4:
            return state

        # Bağlı bileşen tahmini (mesafe bazlı kümeleme)
        distances = squareform(pdist(points))
        threshold = np.percentile(distances, 30)  # %30 mesafe eşiği

        # Basit connected components (DFS)
        n = len(points)
        visited = [False] * n
        components = 0

        def dfs(node):
            visited[node] = True
            for j in range(n):
                if not visited[j] and distances[node][j] < threshold:
                    dfs(j)

        for i in range(n):
            if not visited[i]:
                dfs(i)
                components += 1

        state.h0_components = components

        # Delik tahmini (Delaunay triangulation boşlukları)
        try:
            tri = Delaunay(points)
            simplices = tri.simplices
            # Büyük üçgenler = delik
            large_triangles = 0
            for simplex in simplices:
                triangle_pts = points[simplex]
                area = 0.5 * abs(
                    (triangle_pts[1][0] - triangle_pts[0][0]) *
                    (triangle_pts[2][1] - triangle_pts[0][1]) -
                    (triangle_pts[2][0] - triangle_pts[0][0]) *
                    (triangle_pts[1][1] - triangle_pts[0][1])
                )
                if area > 100:  # 100 m^2 üstü = büyük boşluk
                    large_triangles += 1

            state.h1_holes = large_triangles
        except Exception:
            pass

        # Persistence entropy tahmini
        if len(points) > 3:
            dists = pdist(points)
            hist, _ = np.histogram(dists, bins=10, density=True)
            hist = hist[hist > 0]
            state.persistence_entropy = round(
                float(-np.sum(hist * np.log2(hist + 1e-10))), 4,
            )

        return state

    def _manual_pdist(self, points: np.ndarray) -> np.ndarray:
        """scipy olmadan pairwise distance."""
        n = len(points)
        dists = []
        for i in range(n):
            for j in range(i + 1, n):
                d = np.sqrt(np.sum((points[i] - points[j]) ** 2))
                dists.append(d)
        return np.array(dists)

    # ═══════════════════════════════════════════
    #  BÜTÜNLÜK SKORU
    # ═══════════════════════════════════════════
    def _calculate_integrity(self, state: FormationState) -> float:
        """Formasyon bütünlüğü skoru hesapla (0-1)."""
        score = 1.0

        # Bağlı bileşen sayısı (1 = ideal, 3+ = kötü)
        if state.h0_components > 2:
            score -= 0.3 * (state.h0_components - 1)
        elif state.h0_components > 1:
            score -= 0.1

        # Delik sayısı (0 = ideal, 3+ = kötü)
        if state.h1_holes > 2:
            score -= 0.25
        elif state.h1_holes > 0:
            score -= 0.1

        # Aşırı açılma
        if state.convex_hull_area > self.STRETCHED_THRESHOLD:
            ratio = state.convex_hull_area / self.STRETCHED_THRESHOLD
            score -= min(0.3, 0.15 * (ratio - 1))

        # Sıkılık düşükse
        if state.compactness < 0.3:
            score -= 0.2

        return round(max(0.0, min(1.0, score)), 3)

    # ═══════════════════════════════════════════
    #  UYARI SİSTEMİ
    # ═══════════════════════════════════════════
    def check_integrity(self, coords: list[tuple[float, float]],
                          team: str = "",
                          match_id: str = "") -> list[FormationAlert]:
        """Formasyon bozulma kontrolü."""
        state = self.analyze_formation(coords, team)
        alerts: list[FormationAlert] = []

        # Panik seviyesi
        if state.formation_integrity < self.PANIC_THRESHOLD:
            alerts.append(FormationAlert(
                team=team, match_id=match_id,
                alert_type="panic",
                severity="critical",
                integrity=state.formation_integrity,
                description=(
                    f"FORMASYON PANİĞİ: {team} tamamen dağılmış! "
                    f"Bütünlük: {state.formation_integrity:.0%}, "
                    f"Bağlı bileşen: {state.h0_components}, "
                    f"Boşluk: {state.h1_holes}"
                ),
                details={"state": state.__dict__},
            ))

        # Çöküş
        elif state.formation_integrity < self.INTEGRITY_ALERT:
            alerts.append(FormationAlert(
                team=team, match_id=match_id,
                alert_type="collapse",
                severity="high",
                integrity=state.formation_integrity,
                description=(
                    f"SAVUNMA ÇÖKÜYOR: {team} formasyon bütünlüğü "
                    f"{state.formation_integrity:.0%}. "
                    f"Alan: {state.convex_hull_area:.0f}m^2"
                ),
            ))

        # Büyük boşluk
        if state.h1_holes > 2:
            alerts.append(FormationAlert(
                team=team, match_id=match_id,
                alert_type="gap",
                severity="high",
                integrity=state.formation_integrity,
                description=(
                    f"SAVUNMA BOŞLUĞU: {team} hattında "
                    f"{state.h1_holes} büyük boşluk tespit edildi!"
                ),
            ))

        # Aşırı açılma
        if state.convex_hull_area > self.STRETCHED_THRESHOLD * 1.3:
            alerts.append(FormationAlert(
                team=team, match_id=match_id,
                alert_type="stretch",
                severity="medium",
                integrity=state.formation_integrity,
                description=(
                    f"AŞIRI AÇILMA: {team} çok geniş oynuyor "
                    f"(alan={state.convex_hull_area:.0f}m^2). "
                    f"Kontra atak riski yüksek."
                ),
            ))

        return alerts

    # ═══════════════════════════════════════════
    #  KARŞILAŞTIRMA
    # ═══════════════════════════════════════════
    def compare(self, before: FormationState,
                 after: FormationState) -> FormationComparison:
        """İki formasyon durumunu karşılaştır."""
        comp = FormationComparison(
            team=after.team,
            state_before=before,
            state_after=after,
        )

        comp.integrity_change = round(
            after.formation_integrity - before.formation_integrity, 3,
        )

        if before.convex_hull_area > 0:
            comp.area_change_pct = round(
                (after.convex_hull_area - before.convex_hull_area)
                / before.convex_hull_area, 3,
            )

        comp.compactness_change = round(
            after.compactness - before.compactness, 3,
        )

        comp.is_deteriorating = comp.integrity_change < -0.1

        # Yorumlama
        if comp.is_deteriorating:
            comp.interpretation = (
                f"{after.team} formasyon kaybediyor! "
                f"Bütünlük: {before.formation_integrity:.0%} → "
                f"{after.formation_integrity:.0%} "
                f"({comp.integrity_change:+.0%})"
            )
        elif comp.integrity_change > 0.1:
            comp.interpretation = (
                f"{after.team} formasyonunu toparlıyor "
                f"({comp.integrity_change:+.0%})"
            )
        else:
            comp.interpretation = f"{after.team} formasyon stabil."

        return comp

    def get_trend(self, team: str, window: int = 10) -> dict:
        """Formasyonun son N karedeki trendi."""
        history = self._history.get(team, [])
        if len(history) < 2:
            return {"trend": "unknown", "team": team}

        recent = history[-window:]
        integrities = [s.formation_integrity for s in recent]

        trend_val = float(np.polyfit(range(len(integrities)), integrities, 1)[0])

        return {
            "team": team,
            "trend": "improving" if trend_val > 0.01 else (
                "deteriorating" if trend_val < -0.01 else "stable"
            ),
            "trend_slope": round(trend_val, 4),
            "current_integrity": round(integrities[-1], 3),
            "avg_integrity": round(float(np.mean(integrities)), 3),
            "min_integrity": round(float(np.min(integrities)), 3),
            "n_frames": len(recent),
            "method": recent[-1].method if recent else "unknown",
        }
