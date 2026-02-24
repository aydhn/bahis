"""
topology_mapper.py – Topological Mapper (Kepler Mapper).

Veriyi Excel tablosu gibi değil, bir "Galaksi Haritası" gibi
görselleştirir. Benzer maçlar birer yıldız kümesi gibi birleşir.

Kavramlar:
  - Kepler Mapper (kmapper): Mapper algoritmasının Python uygulaması
  - Lens Function: Veriyi düşük boyuta yansıtan fonksiyon (PCA, t-SNE)
  - Cover: Lens uzayını örtecek aralıklar (intervals + overlap)
  - Nerve: Her aralıktaki verinin kümelenmesi → simplisyel kompleks
  - Cluster: Her düğüm bir maç kümesi, bağlantılar ortak maçlar
  - Anomalous Cluster: Şüpheli/anormal maçların toplandığı küme
  - HTML Visualization: İnteraktif ağ görselleştirmesi

Akış:
  1. Maç özellik vektörlerini yükle (xG, şut, oran, vb.)
  2. Lens fonksiyonu uygula (PCA 1. bileşen veya isolation score)
  3. Cover + kümeleme → Mapper grafiği oluştur
  4. Her düğüme renk ver (ortalama sonuç veya anomali skoru)
  5. Bugünkü maçı grafa yansıt → hangi kümeye düşüyor?
  6. Anomali kümesine yakınsa → uyarı ver

Teknoloji: kmapper (KeplerMapper)
Fallback: PCA + KMeans + NetworkX
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

try:
    import kmapper as km
    KMAPPER_OK = True
except ImportError:
    KMAPPER_OK = False
    logger.debug("kmapper yüklü değil – PCA+KMeans fallback.")

try:
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import NearestNeighbors
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

try:
    import networkx as nx
    NX_OK = True
except ImportError:
    NX_OK = False

ROOT = Path(__file__).resolve().parent.parent.parent
VIZ_DIR = ROOT / "data" / "topology"
VIZ_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════
#  VERİ YAPILARI
# ═══════════════════════════════════════════════
@dataclass
class MapperNode:
    """Mapper grafik düğümü."""
    node_id: str = ""
    n_members: int = 0         # Bu kümedeki maç sayısı
    member_indices: list[int] = field(default_factory=list)
    avg_features: dict[str, float] = field(default_factory=dict)
    label: str = ""            # "normal" | "anomalous" | "high_risk"
    color_value: float = 0.0   # Renklendirme değeri


@dataclass
class MapperReport:
    """Topolojik harita raporu."""
    match_id: str = ""
    team: str = ""
    # Mapper
    n_nodes: int = 0           # Grafikteki toplam düğüm sayısı
    n_edges: int = 0           # Toplam bağlantı sayısı
    n_clusters: int = 0        # Küme sayısı
    # Maçın durumu
    assigned_cluster: int = -1
    cluster_label: str = ""
    cluster_size: int = 0
    cluster_avg_outcome: float = 0.0  # Kümedeki maçların ortalama sonucu
    nearest_neighbors: list[int] = field(default_factory=list)
    # Anomali
    is_anomalous: bool = False
    anomaly_score: float = 0.0
    anomalous_clusters: list[int] = field(default_factory=list)
    # Çıktı
    html_path: str = ""        # İnteraktif HTML dosyası
    method: str = ""
    recommendation: str = ""


# ═══════════════════════════════════════════════
#  PCA + KMEANS FALLBACK
# ═══════════════════════════════════════════════
def fallback_mapper(X: np.ndarray, n_clusters: int = 8) -> dict:
    """PCA + KMeans ile basit topolojik harita."""
    result = {
        "labels": np.zeros(len(X), dtype=int),
        "centers": np.zeros((n_clusters, X.shape[1])),
        "pca_2d": np.zeros((len(X), 2)),
        "graph": None,
    }

    if not SKLEARN_OK:
        return result

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA
    n_comp = min(2, X.shape[1])
    pca = PCA(n_components=n_comp)
    X_pca = pca.fit_transform(X_scaled)
    result["pca_2d"] = X_pca

    # KMeans
    k = min(n_clusters, len(X))
    km_model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km_model.fit_predict(X_scaled)
    result["labels"] = labels
    result["centers"] = km_model.cluster_centers_

    # NetworkX grafik
    if NX_OK:
        G = nx.Graph()
        for i in range(k):
            members = np.where(labels == i)[0].tolist()
            G.add_node(i, size=len(members), members=members)

        # Ortak komşuları olan kümeleri bağla
        nn = NearestNeighbors(n_neighbors=min(5, len(X)))
        nn.fit(X_scaled)
        for i in range(k):
            center_i = km_model.cluster_centers_[i].reshape(1, -1)
            dists, indices = nn.kneighbors(center_i)
            neighbor_labels = set(labels[indices[0]])
            for j in neighbor_labels:
                if j != i and not G.has_edge(i, j):
                    G.add_edge(i, j)

        result["graph"] = G

    return result


# ═══════════════════════════════════════════════
#  TOPOLOGY MAPPER (Ana Sınıf)
# ═══════════════════════════════════════════════
class TopologyMapper:
    """Kepler Mapper ile topolojik veri analizi.

    Kullanım:
        tm = TopologyMapper()

        # Maç verisini yükle (N maç, M özellik)
        tm.fit(X_historical, labels=y_labels)

        # Bugünkü maçı analiz et
        report = tm.analyze_match(X_today, match_id="gs_fb")

        # HTML görselleştirmesi
        tm.visualize("mapper_report.html")
    """

    def __init__(self, n_cubes: int = 10, overlap: float = 0.3,
                 n_clusters_per_cube: int = 3):
        self._n_cubes = n_cubes
        self._overlap = overlap
        self._n_clusters = n_clusters_per_cube
        self._mapper: Any = None
        self._graph: Any = None
        self._X: np.ndarray | None = None
        self._labels: np.ndarray | None = None
        self._scaler: Any = None
        self._lens: np.ndarray | None = None
        self._simplicial_complex: Any = None
        # Fallback
        self._fallback_result: dict | None = None

        logger.debug(
            f"[Mapper] Başlatıldı: cubes={n_cubes}, "
            f"overlap={overlap}, clusters/cube={n_clusters_per_cube}"
        )

    def fit(self, X: np.ndarray, labels: np.ndarray | None = None,
            column_names: list[str] | None = None) -> None:
        """Mapper grafiğini oluştur."""
        X = np.array(X, dtype=np.float64)
        mask = ~np.isnan(X).any(axis=1)
        X = X[mask]
        self._X = X

        if labels is not None:
            self._labels = np.array(labels)[mask]

        if SKLEARN_OK:
            self._scaler = StandardScaler()
            X_scaled = self._scaler.fit_transform(X)
        else:
            X_scaled = X

        if KMAPPER_OK:
            try:
                self._mapper = km.KeplerMapper(verbose=0)
                self._lens = self._mapper.fit_transform(
                    X_scaled,
                    projection=[0],  # PCA 1. bileşen
                )
                self._simplicial_complex = self._mapper.map(
                    self._lens,
                    X_scaled,
                    cover=km.Cover(
                        n_cubes=self._n_cubes,
                        perc_overlap=self._overlap,
                    ),
                    clusterer=KMeans(
                        n_clusters=self._n_clusters,
                        random_state=42,
                        n_init=10,
                    ) if SKLEARN_OK else None,
                )
                logger.info(
                    f"[Mapper] KeplerMapper fit: {len(X)} maç, "
                    f"{len(self._simplicial_complex.get('nodes', {}))} düğüm"
                )
                return
            except Exception as e:
                logger.debug(f"[Mapper] KeplerMapper hatası: {e}")

        # Fallback
        self._fallback_result = fallback_mapper(
            X, n_clusters=self._n_cubes,
        )
        logger.info(
            f"[Mapper] Fallback fit: {len(X)} maç, "
            f"{self._n_cubes} küme"
        )

    def analyze_match(self, features: np.ndarray | list,
                        match_id: str = "",
                        team: str = "",
                        outcome_column: int = -1) -> MapperReport:
        """Tek bir maçı topolojik haritada konumlandır."""
        report = MapperReport(match_id=match_id, team=team)
        x = np.array(features, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)

        if self._X is None:
            report.recommendation = "Mapper henüz eğitilmedi."
            report.method = "not_fitted"
            return report

        # Normalize
        if self._scaler:
            x_scaled = self._scaler.transform(x)
        else:
            x_scaled = x

        # KeplerMapper yolu
        if KMAPPER_OK and self._simplicial_complex:
            report = self._analyze_kmapper(x_scaled, report)
            report.method = "kepler_mapper"
        elif self._fallback_result:
            report = self._analyze_fallback(x_scaled, report)
            report.method = "pca_kmeans_fallback"
        else:
            report.recommendation = "Mapper verisi yok."
            report.method = "none"
            return report

        # Anomali kontrolü
        report = self._check_anomaly(x_scaled, report)
        report.recommendation = self._advice(report)
        return report

    def _analyze_kmapper(self, x_scaled: np.ndarray,
                            report: MapperReport) -> MapperReport:
        """KeplerMapper ile analiz."""
        sc = self._simplicial_complex
        nodes = sc.get("nodes", {})
        links = sc.get("links", {})

        report.n_nodes = len(nodes)
        report.n_edges = sum(len(v) for v in links.values())
        report.n_clusters = len(nodes)

        # En yakın düğümü bul
        min_dist = float("inf")
        best_node = None

        for node_id, member_indices in nodes.items():
            if not member_indices:
                continue
            if self._X is not None:
                center = np.mean(self._X[member_indices], axis=0)
                if self._scaler:
                    center = self._scaler.transform(center.reshape(1, -1))[0]
                dist = float(np.linalg.norm(x_scaled[0] - center))
                if dist < min_dist:
                    min_dist = dist
                    best_node = node_id
                    report.cluster_size = len(member_indices)
                    report.nearest_neighbors = member_indices[:10]

        if best_node is not None:
            report.assigned_cluster = hash(best_node) % 1000
            members = nodes[best_node]
            if self._labels is not None and len(members) > 0:
                valid_members = [m for m in members if m < len(self._labels)]
                if valid_members:
                    report.cluster_avg_outcome = round(
                        float(np.mean(self._labels[valid_members])), 3,
                    )

        return report

    def _analyze_fallback(self, x_scaled: np.ndarray,
                             report: MapperReport) -> MapperReport:
        """Fallback (PCA+KMeans) ile analiz."""
        fb = self._fallback_result
        if fb is None:
            return report

        labels = fb["labels"]
        centers = fb["centers"]
        graph = fb["graph"]

        report.n_clusters = len(np.unique(labels))
        if graph:
            report.n_nodes = graph.number_of_nodes()
            report.n_edges = graph.number_of_edges()

        # En yakın merkeze ata
        dists = np.linalg.norm(centers - x_scaled[0], axis=1)
        cluster_id = int(np.argmin(dists))
        report.assigned_cluster = cluster_id

        members = np.where(labels == cluster_id)[0]
        report.cluster_size = len(members)
        report.nearest_neighbors = members[:10].tolist()

        if self._labels is not None:
            valid = [m for m in members if m < len(self._labels)]
            if valid:
                report.cluster_avg_outcome = round(
                    float(np.mean(self._labels[valid])), 3,
                )

        # Anomali kümeleri (küçük veya uç kümeler)
        for c in range(report.n_clusters):
            c_members = np.where(labels == c)[0]
            if len(c_members) < max(3, len(labels) * 0.05):
                report.anomalous_clusters.append(c)

        return report

    def _check_anomaly(self, x_scaled: np.ndarray,
                         report: MapperReport) -> MapperReport:
        """Anomali kontrolü."""
        if self._X is None:
            return report

        # KNN uzaklığı
        if SKLEARN_OK and len(self._X) > 5:
            try:
                X_scaled_all = (
                    self._scaler.transform(self._X)
                    if self._scaler else self._X
                )
                nn = NearestNeighbors(n_neighbors=min(5, len(self._X)))
                nn.fit(X_scaled_all)
                dists, _ = nn.kneighbors(x_scaled)
                avg_dist = float(np.mean(dists[0]))

                # Tüm veri için ortalama uzaklık
                all_dists, _ = nn.kneighbors(X_scaled_all)
                global_avg = float(np.mean(all_dists))
                global_std = float(np.std(np.mean(all_dists, axis=1)))

                z_score = (avg_dist - global_avg) / max(global_std, 1e-6)
                report.anomaly_score = round(float(min(1.0, max(0, z_score / 3))), 3)

                if z_score > 2.0:
                    report.is_anomalous = True
                    report.cluster_label = "anomalous"
                elif report.assigned_cluster in report.anomalous_clusters:
                    report.is_anomalous = True
                    report.cluster_label = "high_risk"
                else:
                    report.cluster_label = "normal"
            except Exception:
                report.cluster_label = "unknown"
        else:
            report.cluster_label = "unknown"

        return report

    def visualize(self, filename: str = "mapper_graph.html") -> str:
        """HTML görselleştirmesi oluştur."""
        path = VIZ_DIR / filename

        if KMAPPER_OK and self._mapper and self._simplicial_complex:
            try:
                color_fn = None
                if self._labels is not None:
                    color_fn = self._labels.astype(float)

                html = self._mapper.visualize(
                    self._simplicial_complex,
                    path_html=str(path),
                    color_values=color_fn,
                    title="Maç Topolojik Haritası",
                )
                logger.info(f"[Mapper] HTML kaydedildi: {path}")
                return str(path)
            except Exception as e:
                logger.debug(f"[Mapper] Visualize hatası: {e}")

        # Fallback: basit bilgi dosyası
        info = {
            "method": "fallback",
            "n_clusters": self._n_cubes,
            "data_points": len(self._X) if self._X is not None else 0,
        }
        path.write_text(str(info), encoding="utf-8")
        return str(path)

    def _advice(self, r: MapperReport) -> str:
        if r.is_anomalous:
            return (
                f"ANOMALİ: Bu maç topolojik haritada anormal bölgede! "
                f"Anomali skoru: {r.anomaly_score:.0%}. "
                f"Küme: #{r.assigned_cluster} ({r.cluster_size} maç). "
                f"Şüpheli hareket veya şike riski. DİKKAT!"
            )
        if r.cluster_label == "high_risk":
            return (
                f"YÜKSEK RİSK: Küme #{r.assigned_cluster} az üyeli "
                f"({r.cluster_size}). Tarihsel sonuç: {r.cluster_avg_outcome:.2f}. "
                f"Stake düşür."
            )
        return (
            f"NORMAL: Küme #{r.assigned_cluster} ({r.cluster_size} maç). "
            f"Tarihsel sonuç: {r.cluster_avg_outcome:.2f}. "
            f"Güvenli bölge."
        )
