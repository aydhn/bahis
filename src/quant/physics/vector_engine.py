"""
vector_engine.py – Vektör Veritabanı ve Benzerlik Araması (Tarihsel İkizler).

Regresyon bazen bağlamı kaçırır. Biz maçları 128 boyutlu
"Özet Vektörüne" (Embedding) dönüştürüp, uzaydaki en yakın
komşusunu arayacağız.

Süreç:
  1. Her maçı (takım gücü, hava, hakem, form...) → 128D vektör
  2. Bugünkü maçı vektöre çevir
  3. FAISS ile "matematiksel en benzer" geçmiş 50 maçı bul
  4. Bu 50 maçın %80'i "Üst" bitmişse → bugünkü maç da "Üst"

Teknoloji: FAISS (Facebook AI Similarity Search)
Fallback: scikit-learn NearestNeighbors (FAISS yoksa)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from loguru import logger

try:
    import faiss
    FAISS_OK = True
except ImportError:
    FAISS_OK = False
    logger.info("faiss yüklü değil – sklearn NearestNeighbors fallback.")

try:
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

ROOT = Path(__file__).resolve().parent.parent.parent
INDEX_DIR = ROOT / "models" / "faiss"
INDEX_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════
#  FEATURE → EMBEDDING
# ═══════════════════════════════════════════════
FEATURE_KEYS = [
    "home_xg", "away_xg", "home_goals", "away_goals",
    "home_possession", "away_possession",
    "home_shots", "away_shots",
    "home_shots_on_target", "away_shots_on_target",
    "home_corners", "away_corners",
    "home_yellows", "away_yellows",
    "home_reds", "away_reds",
    "home_form_pts", "away_form_pts",        # Son 5 maç puan
    "home_odds", "draw_odds", "away_odds",
    "home_elo", "away_elo",
    "home_strength", "away_strength",        # Kalman
    "home_momentum", "away_momentum",
    "temperature", "humidity",               # Hava durumu
    "travel_distance",                       # Deplasman mesafesi
    "rest_days_home", "rest_days_away",
    "referee_card_avg",                      # Hakem kart ort.
]


@dataclass
class SimilarMatch:
    """Benzer maç sonucu."""
    match_id: str = ""
    distance: float = 0.0
    similarity: float = 0.0     # 0-1 (1 = birebir aynı)
    home_team: str = ""
    away_team: str = ""
    date: str = ""
    home_goals: int = 0
    away_goals: int = 0
    result: str = ""            # H / D / A
    total_goals: int = 0
    features: dict = field(default_factory=dict)


@dataclass
class SimilarityReport:
    """Benzerlik analizi raporu."""
    query_match_id: str = ""
    k: int = 50
    similar_matches: list[SimilarMatch] = field(default_factory=list)
    distribution: dict = field(default_factory=dict)  # {"H": 24, "D": 12, "A": 14}
    suggested_result: str = ""
    suggested_confidence: float = 0.0
    over_25_rate: float = 0.0
    btts_rate: float = 0.0
    avg_total_goals: float = 0.0
    method: str = "faiss"


class VectorMatchEngine:
    """FAISS ile maç benzerlik motoru.

    Kullanım:
        engine = VectorMatchEngine(dim=32)
        # Geçmiş maçları indeksle
        engine.index_matches(historical_data)
        # Benzer maçları bul
        report = engine.find_similar(current_match_features, k=50)
    """

    def __init__(self, dim: int = 32):
        self._dim = dim
        self._index = None
        self._scaler = None
        self._match_ids: list[str] = []
        self._match_metadata: list[dict] = []
        self._raw_features: list[np.ndarray] = []
        self._n_indexed = 0
        logger.debug(f"VectorMatchEngine başlatıldı (dim={dim}).")

    # ═══════════════════════════════════════════
    #  EMBEDDING OLUŞTURMA
    # ═══════════════════════════════════════════
    def _to_vector(self, match: dict) -> np.ndarray:
        """Maç verisini sabit boyutlu vektöre dönüştür."""
        values = []
        for key in FEATURE_KEYS:
            v = match.get(key, 0)
            try:
                values.append(float(v) if v is not None else 0.0)
            except (ValueError, TypeError):
                values.append(0.0)

        vec = np.array(values, dtype=np.float32)

        # Boyutu ayarla: padding veya truncation
        if len(vec) < self._dim:
            vec = np.pad(vec, (0, self._dim - len(vec)), constant_values=0)
        elif len(vec) > self._dim:
            vec = vec[:self._dim]

        return vec

    # ═══════════════════════════════════════════
    #  İNDEKSLEME
    # ═══════════════════════════════════════════
    def index_matches(self, matches: list[dict]) -> int:
        """Geçmiş maçları FAISS indeksine yükle."""
        if not matches:
            return 0

        vectors = []
        for m in matches:
            vec = self._to_vector(m)
            vectors.append(vec)
            self._match_ids.append(m.get("match_id", str(len(self._match_ids))))
            self._match_metadata.append(m)

        matrix = np.vstack(vectors).astype(np.float32)

        # Normalize (StandardScaler)
        if SKLEARN_OK:
            self._scaler = StandardScaler()
            matrix = self._scaler.fit_transform(matrix).astype(np.float32)

        self._raw_features = [matrix[i] for i in range(matrix.shape[0])]

        # FAISS indeksi oluştur
        if FAISS_OK:
            self._index = faiss.IndexFlatL2(self._dim)
            self._index.add(matrix)
            method = "faiss"
        elif SKLEARN_OK:
            self._index = NearestNeighbors(
                n_neighbors=min(50, len(matches)),
                metric="euclidean", algorithm="ball_tree",
            )
            self._index.fit(matrix)
            method = "sklearn"
        else:
            self._index = matrix  # Raw numpy fallback
            method = "numpy"

        self._n_indexed = len(matches)
        logger.info(
            f"[Vector] {len(matches)} maç indekslendi "
            f"(dim={self._dim}, method={method})"
        )
        return len(matches)

    def add_match(self, match: dict):
        """Tek maç ekle (artımlı)."""
        vec = self._to_vector(match)

        if self._scaler:
            vec = self._scaler.transform(vec.reshape(1, -1)).astype(np.float32)[0]

        self._match_ids.append(match.get("match_id", ""))
        self._match_metadata.append(match)
        self._raw_features.append(vec)

        if FAISS_OK and self._index is not None:
            self._index.add(vec.reshape(1, -1))
        elif SKLEARN_OK and self._index is not None and hasattr(self._index, "fit"):
            # sklearn'de artımlı ekleme yok, yeniden fit
            matrix = np.vstack(self._raw_features).astype(np.float32)
            self._index.fit(matrix)

        self._n_indexed += 1

    # ═══════════════════════════════════════════
    #  BENZERLİK ARAŞTIRMASI
    # ═══════════════════════════════════════════
    def find_similar(self, query: dict, k: int = 50) -> SimilarityReport:
        """Verilen maça en benzer k maçı bul."""
        if self._n_indexed == 0:
            return SimilarityReport(
                query_match_id=query.get("match_id", ""),
                k=k, method="empty",
            )

        query_vec = self._to_vector(query)
        if self._scaler:
            query_vec = self._scaler.transform(
                query_vec.reshape(1, -1)
            ).astype(np.float32)[0]

        k = min(k, self._n_indexed)

        if FAISS_OK and isinstance(self._index, faiss.Index):
            distances, indices = self._index.search(
                query_vec.reshape(1, -1), k,
            )
            distances = distances[0]
            indices = indices[0]
            method = "faiss"
        elif SKLEARN_OK and hasattr(self._index, "kneighbors"):
            distances, indices = self._index.kneighbors(
                query_vec.reshape(1, -1), n_neighbors=k,
            )
            distances = distances[0]
            indices = indices[0]
            method = "sklearn"
        else:
            # Numpy brute-force
            matrix = np.vstack(self._raw_features).astype(np.float32)
            dists = np.linalg.norm(matrix - query_vec, axis=1)
            indices = np.argsort(dists)[:k]
            distances = dists[indices]
            method = "numpy"

        # Sonuçları oluştur
        similar = []
        for dist, idx in zip(distances, indices):
            idx = int(idx)
            if idx < 0 or idx >= len(self._match_metadata):
                continue

            meta = self._match_metadata[idx]
            max_dist = float(np.max(distances)) if np.max(distances) > 0 else 1.0
            similarity = max(0, 1.0 - float(dist) / max_dist)

            hg = int(meta.get("home_goals", 0))
            ag = int(meta.get("away_goals", 0))
            result = "H" if hg > ag else ("D" if hg == ag else "A")

            similar.append(SimilarMatch(
                match_id=self._match_ids[idx],
                distance=float(dist),
                similarity=round(similarity, 3),
                home_team=meta.get("home_team", ""),
                away_team=meta.get("away_team", ""),
                date=meta.get("date", ""),
                home_goals=hg,
                away_goals=ag,
                result=result,
                total_goals=hg + ag,
            ))

        # Dağılım analizi
        distribution = {"H": 0, "D": 0, "A": 0}
        total_goals_list = []
        btts_count = 0

        for m in similar:
            distribution[m.result] = distribution.get(m.result, 0) + 1
            total_goals_list.append(m.total_goals)
            if m.home_goals > 0 and m.away_goals > 0:
                btts_count += 1

        n = len(similar) or 1
        suggested = max(distribution, key=distribution.get)
        confidence = distribution[suggested] / n

        over_25 = sum(1 for g in total_goals_list if g > 2) / n
        btts_rate = btts_count / n
        avg_goals = float(np.mean(total_goals_list)) if total_goals_list else 0

        return SimilarityReport(
            query_match_id=query.get("match_id", ""),
            k=k,
            similar_matches=similar,
            distribution=distribution,
            suggested_result=suggested,
            suggested_confidence=round(confidence, 3),
            over_25_rate=round(over_25, 3),
            btts_rate=round(btts_rate, 3),
            avg_total_goals=round(avg_goals, 2),
            method=method,
        )

    # ═══════════════════════════════════════════
    #  İNDEKS KAYDET / YÜKLE
    # ═══════════════════════════════════════════
    def save_index(self, name: str = "match_index"):
        """FAISS indeksini diske kaydet."""
        if FAISS_OK and isinstance(self._index, faiss.Index):
            path = INDEX_DIR / f"{name}.faiss"
            faiss.write_index(self._index, str(path))
            logger.info(f"[Vector] İndeks kaydedildi: {path}")

        # Metadata kaydet
        import json
        meta_path = INDEX_DIR / f"{name}_meta.json"
        meta_path.write_text(json.dumps({
            "match_ids": self._match_ids,
            "n_indexed": self._n_indexed,
            "dim": self._dim,
        }, ensure_ascii=False))

    def load_index(self, name: str = "match_index") -> bool:
        """FAISS indeksini diskten yükle."""
        if FAISS_OK:
            path = INDEX_DIR / f"{name}.faiss"
            if path.exists():
                self._index = faiss.read_index(str(path))
                logger.info(f"[Vector] İndeks yüklendi: {path}")

                import json
                meta_path = INDEX_DIR / f"{name}_meta.json"
                if meta_path.exists():
                    meta = json.loads(meta_path.read_text())
                    self._match_ids = meta.get("match_ids", [])
                    self._n_indexed = meta.get("n_indexed", 0)
                return True
        return False

    @property
    def n_indexed(self) -> int:
        return self._n_indexed
