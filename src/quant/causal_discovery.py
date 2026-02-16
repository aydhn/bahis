"""
causal_discovery.py – Nedensel Bağlantı Keşfi (Causal Discovery).

Korelasyon kumardır. Nedensellik bilimdir. Bu modül istatistiklerin
içindeki "Gizli Nedensellik Ağını" (DAG) çıkarır.

Kavramlar:
  - DAG (Directed Acyclic Graph): Değişkenler arası yönlü döngüsüz grafik
  - PC Algorithm: Constraint-based nedensellik keşfi
    1. Tam bağlı grafik ile başla
    2. Koşullu bağımsızlık testleri ile kenarları ele
    3. v-yapıları (colliders) tespit ederek yönleri belirle
  - Conditional Independence: P(X⊥Y|Z) — X ve Y, Z bilindiğinde bağımsız mı?
  - Partial Correlation: Diğer değişkenler kontrol edildikten sonra korelasyon
  - Fisher's Z-test: Kısmi korelasyonun anlamlılığını test eder
  - Granger Causality: Zaman serilerinde "X, Y'yi öngörüyor mu?"
  - Intervention Analysis: "Kırmızı kart olmasaydı ne olurdu?" (Counterfactual)
  - Skeleton: Yönsüz kenar yapısı (PC'nin 1. aşaması)
  - Orientation: v-yapıları + Meek kuralları ile yönlendirme (2. aşama)
  - Separation Set: İki değişkeni ayıran koşullandırma kümesi

Akış:
  1. Maç verileri (xG, shots, possession, corners, goals, ...) alınır
  2. PC algoritması ile nedensellik DAG'ı çıkarılır
  3. Granger testi ile zaman serisi nedenselliği kontrol edilir
  4. Kök nedenler (root causes) belirlenir
  5. Ters nedensellik (reverse causation) tuzakları tespit edilir
  6. Bahis onay/red sinyali üretilir

Teknoloji: causal-learn (PC Algorithm)
Fallback: scipy + numpy tabanlı PC implementasyonu
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
from loguru import logger

try:
    from scipy import stats as sp_stats
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False

try:
    from causallearn.search.ConstraintBased.PC import pc as cl_pc
    from causallearn.utils.cit import fisherz
    CAUSALLEARN_OK = True
except ImportError:
    CAUSALLEARN_OK = False
    logger.debug("causal-learn yüklü değil – manuel PC fallback.")


# ═══════════════════════════════════════════════
#  VERİ YAPILARI
# ═══════════════════════════════════════════════
@dataclass
class CausalEdge:
    """Nedensel kenar."""
    source: str                   # Neden
    target: str                   # Sonuç
    strength: float = 0.0         # Etki büyüklüğü [0, 1]
    p_value: float = 1.0          # İstatistiksel anlamlılık
    edge_type: str = "direct"     # "direct" | "granger" | "collider"
    direction_confidence: float = 0.0  # Yön güveni [0, 1]
    separation_set: list[str] = field(default_factory=list)


@dataclass
class CausalDAG:
    """Nedensellik DAG raporu."""
    edges: list[CausalEdge] = field(default_factory=list)
    root_causes: list[str] = field(default_factory=list)
    leaf_effects: list[str] = field(default_factory=list)
    colliders: list[tuple[str, str, str]] = field(default_factory=list)
    n_variables: int = 0
    n_edges: int = 0
    density: float = 0.0
    method: str = ""

    def get_causes_of(self, target: str) -> list[CausalEdge]:
        return [e for e in self.edges if e.target == target]

    def get_effects_of(self, source: str) -> list[CausalEdge]:
        return [e for e in self.edges if e.source == source]


@dataclass
class CausalReport:
    """Maç nedensellik analiz raporu."""
    match_id: str = ""
    dag: CausalDAG = field(default_factory=CausalDAG)
    # Futbol-spesifik bulgular
    goal_root_causes: list[str] = field(default_factory=list)
    spurious_correlations: list[str] = field(default_factory=list)
    reverse_causation_warnings: list[str] = field(default_factory=list)
    # Karar desteği
    causal_confidence: float = 0.0  # [0, 1]
    recommendation: str = ""
    method: str = ""


# ═══════════════════════════════════════════════
#  FİSHER'S Z-TEST (Kısmi Korelasyon Bağımsızlık Testi)
# ═══════════════════════════════════════════════
def _partial_correlation(data: np.ndarray, i: int, j: int,
                          cond_set: list[int]) -> tuple[float, float]:
    """Kısmi korelasyon ve p-değeri hesapla (Fisher's Z-test)."""
    n = data.shape[0]

    if not cond_set:
        if n < 4:
            return 0.0, 1.0
        r = np.corrcoef(data[:, i], data[:, j])[0, 1]
        if np.isnan(r):
            return 0.0, 1.0
        z = 0.5 * np.log((1 + r + 1e-10) / (1 - r + 1e-10))
        stat = abs(z) * np.sqrt(n - 3)
        if SCIPY_OK:
            p = 2 * (1 - sp_stats.norm.cdf(stat))
        else:
            p = 2 * np.exp(-0.5 * stat ** 2) / np.sqrt(2 * np.pi)
        return float(r), float(p)

    # Koşullu kısmi korelasyon
    k = len(cond_set)
    if n - k - 3 < 1:
        return 0.0, 1.0

    try:
        idx = [i, j] + cond_set
        sub = data[:, idx]
        cov = np.cov(sub, rowvar=False)
        if cov.shape[0] < 2:
            return 0.0, 1.0

        prec = np.linalg.inv(cov + np.eye(cov.shape[0]) * 1e-8)
        r = -prec[0, 1] / np.sqrt(abs(prec[0, 0] * prec[1, 1]) + 1e-10)
        r = np.clip(r, -0.999, 0.999)

        z = 0.5 * np.log((1 + r + 1e-10) / (1 - r + 1e-10))
        stat = abs(z) * np.sqrt(n - k - 3)
        if SCIPY_OK:
            p = 2 * (1 - sp_stats.norm.cdf(stat))
        else:
            p = 2 * np.exp(-0.5 * stat ** 2) / np.sqrt(2 * np.pi)
        return float(r), float(p)
    except (np.linalg.LinAlgError, ValueError):
        return 0.0, 1.0


# ═══════════════════════════════════════════════
#  PC ALGORİTMASI (Manuel Implementasyon)
# ═══════════════════════════════════════════════
def _pc_skeleton(data: np.ndarray, alpha: float = 0.05,
                   max_cond_size: int = 3) -> tuple[np.ndarray, dict]:
    """PC algoritmasının iskelet (skeleton) aşaması.

    1. Tam bağlı grafik ile başla
    2. d=0, 1, 2, ... boyutlu koşullandırma kümelerini dene
    3. Koşullu bağımsızlık bulunursa kenarı sil
    """
    p = data.shape[1]
    adj = np.ones((p, p), dtype=bool)
    np.fill_diagonal(adj, False)
    sep_sets: dict[tuple[int, int], list[int]] = {}

    for d in range(max_cond_size + 1):
        for i in range(p):
            neighbors_i = list(np.where(adj[i])[0])
            for j in neighbors_i:
                if not adj[i, j]:
                    continue

                # j'nin komşuları (i hariç)
                possible_cond = [k for k in neighbors_i if k != j]

                if len(possible_cond) < d:
                    continue

                for cond_set in itertools.combinations(possible_cond, d):
                    cond_list = list(cond_set)
                    _, p_val = _partial_correlation(data, i, j, cond_list)

                    if p_val > alpha:
                        adj[i, j] = False
                        adj[j, i] = False
                        sep_sets[(i, j)] = cond_list
                        sep_sets[(j, i)] = cond_list
                        break

    return adj, sep_sets


def _orient_edges(adj: np.ndarray,
                    sep_sets: dict[tuple[int, int], list[int]]) -> np.ndarray:
    """V-yapıları ve Meek kuralları ile kenarları yönlendir.

    Çıktı: dag[i,j]=True ise i→j yönlü kenar var.
    """
    p = adj.shape[0]
    dag = adj.copy()

    # 1. V-yapıları (colliders): i - k - j, k ∉ sep(i,j) → i→k←j
    for k in range(p):
        neighbors = np.where(adj[k])[0]
        for i_idx in range(len(neighbors)):
            for j_idx in range(i_idx + 1, len(neighbors)):
                i, j = neighbors[i_idx], neighbors[j_idx]
                if adj[i, j] or adj[j, i]:
                    continue  # i ve j zaten bağlı, collider değil

                sep = sep_sets.get((i, j), [])
                if k not in sep:
                    # v-yapısı: i→k←j
                    dag[k, i] = False  # k→i yok
                    dag[k, j] = False  # k→j yok
                    # i→k ve j→k kalır

    # 2. Meek kuralları (basitleştirilmiş)
    changed = True
    iterations = 0
    while changed and iterations < 10:
        changed = False
        iterations += 1

        for i in range(p):
            for j in range(p):
                if not dag[i, j] or not dag[j, i]:
                    continue  # Zaten yönlü veya kenar yok

                # Kural 1: i→k→j ise i→j (döngü önleme)
                for k in range(p):
                    if dag[i, k] and not dag[k, i] and dag[k, j]:
                        dag[j, i] = False
                        changed = True

    return dag


# ═══════════════════════════════════════════════
#  CAUSAL DISCOVERY (Ana Sınıf)
# ═══════════════════════════════════════════════
class CausalDiscovery:
    """Nedensellik Keşfi Motoru.

    Kullanım:
        cd = CausalDiscovery(significance=0.05, max_cond_size=3)

        # DAG keşfi
        dag = cd.discover_dag(data_matrix, feature_names)

        # Maç analizi
        report = cd.analyze_match(
            data_matrix, feature_names,
            target="goals", match_id="gs_fb",
        )

        # Kök nedenler
        causes = report.goal_root_causes
    """

    # Bilinen sahte korelasyonlar (ters nedensellik tuzakları)
    KNOWN_SPURIOUS = [
        ("corners", "goals", "Kornerler gol getirmez, baskı göstergesidir"),
        ("fouls", "cards", "Fauller kart getirir ama kart faul getirmez"),
        ("possession", "goals", "Top kontrolü her zaman gol getirmez"),
    ]

    def __init__(self, significance: float = 0.05,
                 max_cond_size: int = 3,
                 min_samples: int = 30,
                 granger_lags: int = 3):
        self._alpha = significance
        self._max_cond = max_cond_size
        self._min_samples = min_samples
        self._granger_lags = granger_lags

        logger.debug(
            f"[CausalDiscovery] Başlatıldı: α={significance}, "
            f"max_cond={max_cond_size}, "
            f"backend={'causal-learn' if CAUSALLEARN_OK else 'manual PC'}"
        )

    def discover_dag(self, data: np.ndarray,
                       feature_names: list[str] | None = None) -> CausalDAG:
        """Nedensellik DAG'ı keşfet."""
        if data.ndim != 2:
            return CausalDAG(method="error: data must be 2D")

        n_samples, n_vars = data.shape
        if n_samples < self._min_samples:
            return CausalDAG(
                method=f"error: insufficient data ({n_samples} < {self._min_samples})",
            )

        if feature_names is None:
            feature_names = [f"X{i}" for i in range(n_vars)]

        # NaN temizliği
        mask = np.all(np.isfinite(data), axis=1)
        clean_data = data[mask]
        if len(clean_data) < self._min_samples:
            return CausalDAG(method="error: too many NaN/Inf")

        # Standardize
        std = np.std(clean_data, axis=0)
        std[std < 1e-8] = 1.0
        standardized = (clean_data - np.mean(clean_data, axis=0)) / std

        if CAUSALLEARN_OK:
            dag = self._discover_causallearn(standardized, feature_names)
        else:
            dag = self._discover_manual(standardized, feature_names)

        return dag

    def _discover_causallearn(self, data: np.ndarray,
                                names: list[str]) -> CausalDAG:
        """causal-learn kütüphanesi ile PC algoritması."""
        try:
            result = cl_pc(
                data,
                alpha=self._alpha,
                indep_test=fisherz,
                stable=True,
                uc_rule=0,
                uc_priority=-1,
            )

            dag = CausalDAG(n_variables=len(names), method="causal-learn PC")
            graph = result.G.graph  # Adjacency matrix

            for i in range(len(names)):
                for j in range(len(names)):
                    if i == j:
                        continue
                    # graph[i,j] == -1 and graph[j,i] == 1 → i→j
                    if graph[i, j] == -1 and graph[j, i] == 1:
                        r, p = _partial_correlation(data, i, j, [])
                        dag.edges.append(CausalEdge(
                            source=names[i],
                            target=names[j],
                            strength=round(abs(r), 4),
                            p_value=round(p, 6),
                            edge_type="direct",
                            direction_confidence=0.8,
                        ))
                    # Yönsüz kenar: graph[i,j] == -1 and graph[j,i] == -1
                    elif (graph[i, j] == -1 and graph[j, i] == -1
                          and i < j):
                        r, p = _partial_correlation(data, i, j, [])
                        dag.edges.append(CausalEdge(
                            source=names[i],
                            target=names[j],
                            strength=round(abs(r), 4),
                            p_value=round(p, 6),
                            edge_type="undirected",
                            direction_confidence=0.3,
                        ))

            dag.n_edges = len(dag.edges)
            max_edges = len(names) * (len(names) - 1) / 2
            dag.density = round(dag.n_edges / max(max_edges, 1), 4)
            dag = self._identify_roles(dag, names)
            return dag

        except Exception as e:
            logger.debug(f"[CausalDiscovery] causal-learn hatası: {e}")
            return self._discover_manual(data, names)

    def _discover_manual(self, data: np.ndarray,
                           names: list[str]) -> CausalDAG:
        """Manuel PC algoritması."""
        dag = CausalDAG(n_variables=len(names), method="manual PC")

        adj, sep_sets = _pc_skeleton(
            data, alpha=self._alpha, max_cond_size=self._max_cond,
        )
        oriented = _orient_edges(adj, sep_sets)

        for i in range(len(names)):
            for j in range(len(names)):
                if i == j:
                    continue
                if oriented[i, j] and not oriented[j, i]:
                    r, p = _partial_correlation(data, i, j, [])
                    sep = sep_sets.get((i, j), [])
                    dag.edges.append(CausalEdge(
                        source=names[i],
                        target=names[j],
                        strength=round(abs(r), 4),
                        p_value=round(p, 6),
                        edge_type="direct",
                        direction_confidence=0.7,
                        separation_set=[names[s] for s in sep if s < len(names)],
                    ))
                elif oriented[i, j] and oriented[j, i] and i < j:
                    r, p = _partial_correlation(data, i, j, [])
                    dag.edges.append(CausalEdge(
                        source=names[i],
                        target=names[j],
                        strength=round(abs(r), 4),
                        p_value=round(p, 6),
                        edge_type="undirected",
                        direction_confidence=0.3,
                    ))

        # Collider tespiti
        for k in range(len(names)):
            parents = [
                e.source for e in dag.edges
                if e.target == names[k] and e.edge_type == "direct"
            ]
            if len(parents) >= 2:
                for p1, p2 in itertools.combinations(parents, 2):
                    dag.colliders.append((p1, names[k], p2))

        dag.n_edges = len(dag.edges)
        max_edges = len(names) * (len(names) - 1) / 2
        dag.density = round(dag.n_edges / max(max_edges, 1), 4)
        dag = self._identify_roles(dag, names)
        return dag

    def _identify_roles(self, dag: CausalDAG,
                          names: list[str]) -> CausalDAG:
        """Kök nedenler ve yaprak etkiler."""
        sources = {e.source for e in dag.edges}
        targets = {e.target for e in dag.edges}

        dag.root_causes = sorted(sources - targets)
        dag.leaf_effects = sorted(targets - sources)
        return dag

    # ─────────────────────────────────────────────
    #  GRANGER NEDENSELLİK
    # ─────────────────────────────────────────────
    def granger_test(self, data: np.ndarray,
                       names: list[str] | None = None) -> list[CausalEdge]:
        """Granger nedensellik testi (zaman serisi)."""
        n_samples, n_vars = data.shape
        if names is None:
            names = [f"X{i}" for i in range(n_vars)]
        if n_samples < self._granger_lags + 5:
            return []

        edges = []
        for i in range(n_vars):
            for j in range(n_vars):
                if i == j:
                    continue

                x = data[:, i]
                y = data[:, j]

                # x lagged → y current korelasyonu
                x_lagged = x[:-(self._granger_lags)]
                y_current = y[self._granger_lags:]

                if len(x_lagged) < 5 or not SCIPY_OK:
                    continue

                r, p = sp_stats.pearsonr(x_lagged, y_current)
                if p < self._alpha and abs(r) > 0.15:
                    edges.append(CausalEdge(
                        source=names[i],
                        target=names[j],
                        strength=round(abs(r), 4),
                        p_value=round(p, 6),
                        edge_type="granger",
                        direction_confidence=0.6,
                    ))

        return edges

    # ─────────────────────────────────────────────
    #  MAÇ ANALİZİ
    # ─────────────────────────────────────────────
    def analyze_match(self, data: np.ndarray,
                        feature_names: list[str],
                        target: str = "goals",
                        match_id: str = "") -> CausalReport:
        """Maç verisi için nedensellik analizi."""
        report = CausalReport(match_id=match_id)

        # 1) DAG keşfi
        dag = self.discover_dag(data, feature_names)
        report.dag = dag
        report.method = dag.method

        # 2) Hedef değişkenin kök nedenleri
        if target in feature_names:
            target_causes = dag.get_causes_of(target)
            report.goal_root_causes = [
                e.source for e in target_causes
                if e.strength > 0.2
            ]

        # 3) Sahte korelasyon tespiti
        for src, tgt, reason in self.KNOWN_SPURIOUS:
            for edge in dag.edges:
                if (edge.source == src and edge.target == tgt
                        and edge.direction_confidence < 0.6):
                    report.spurious_correlations.append(
                        f"{src}→{tgt}: {reason}"
                    )

        # 4) Ters nedensellik uyarıları
        for edge in dag.edges:
            if edge.edge_type == "undirected":
                report.reverse_causation_warnings.append(
                    f"{edge.source}↔{edge.target}: Yön belirsiz "
                    f"(güven={edge.direction_confidence:.0%})"
                )

        # 5) Granger testi
        granger_edges = self.granger_test(data, feature_names)
        dag.edges.extend(granger_edges)

        # 6) Toplam güven
        if dag.edges:
            avg_conf = np.mean([e.direction_confidence for e in dag.edges])
            avg_str = np.mean([e.strength for e in dag.edges])
            report.causal_confidence = round(
                float(avg_conf * 0.6 + avg_str * 0.4), 4,
            )
        else:
            report.causal_confidence = 0.0

        report.recommendation = self._recommendation(report)
        return report

    def _recommendation(self, r: CausalReport) -> str:
        n_causes = len(r.goal_root_causes)
        n_spurious = len(r.spurious_correlations)
        n_reverse = len(r.reverse_causation_warnings)

        if r.causal_confidence < 0.3:
            return (
                f"DÜŞÜK GÜVEN: Nedensellik ağı zayıf (güven={r.causal_confidence:.1%}). "
                f"İstatistiksel sinyallere dikkatli yaklaş."
            )
        if n_spurious > 2:
            return (
                f"UYARI: {n_spurious} sahte korelasyon tespit edildi! "
                f"Model çıktılarını nedensellik filtresinden geçir."
            )
        if n_reverse > 3:
            return (
                f"DİKKAT: {n_reverse} yönü belirsiz ilişki. "
                f"Ters nedensellik riski yüksek."
            )
        if n_causes >= 3:
            return (
                f"GÜÇLÜ: {n_causes} kök neden tespit edildi "
                f"(güven={r.causal_confidence:.1%}). "
                f"Nedensellik ağı tutarlı."
            )
        return f"NORMAL: {r.dag.n_edges} kenar, güven={r.causal_confidence:.1%}."

    # ─────────────────────────────────────────────
    #  GERİYE UYUMLULUK (eski API)
    # ─────────────────────────────────────────────
    def discover(self, features) -> dict:
        """Eski API uyumluluğu."""
        try:
            import polars as pl
            if isinstance(features, pl.DataFrame):
                numeric_cols = [
                    c for c in features.columns
                    if features[c].dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32)
                ]
                if len(numeric_cols) < 2 or features.height < self._min_samples:
                    return {"edges": [], "root_causes": []}

                data = features.select(numeric_cols).to_numpy()
                dag = self.discover_dag(data, numeric_cols)
                return {
                    "edges": [
                        {"from": e.source, "to": e.target,
                         "weight": e.strength, "pvalue": e.p_value}
                        for e in dag.edges
                    ],
                    "root_causes": dag.root_causes,
                    "leaf_effects": dag.leaf_effects,
                    "method": dag.method,
                }
        except Exception:
            pass
        return {"edges": [], "root_causes": []}
