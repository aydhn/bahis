"""
vector_memory.py – Yerel Vektör Arama Motoru (Matches RAG).

Amacı: "Bu maç hangi eski maça benziyor?" sorusunu yanıtlamak.
Teknoloji: Saf NumPy (Ücretsiz, Pinecone/Milvus gerektirmez).

Özellikler:
- Maç istatistiklerini (xG, Şut, Topla Oynama) vektöre çevirir.
- Cosine Similarity ile en benzer K maçı bulur.
- In-memory çalışır, disk üzerine pickle/json ile kaydedilebilir.
"""
import numpy as np
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
from loguru import logger

@dataclass
class MatchVector:
    match_id: str
    vector: List[float]  # [home_xg, away_xg, home_poss, away_poss, ...]
    metadata: Dict[str, Any]

class VectorMemory:
    def __init__(self, storage_path: str = "data/match_vectors.json"):
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.vectors: List[MatchVector] = []
        self._load()

    def _load(self):
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.vectors = [MatchVector(**item) for item in data]
                logger.info(f"[VectorMemory] {len(self.vectors)} vektör yüklendi.")
            except Exception as e:
                logger.error(f"[VectorMemory] Yükleme hatası: {e}")
                self.vectors = []

    def save(self):
        try:
            data = [asdict(v) for v in self.vectors]
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"[VectorMemory] Kayıt hatası: {e}")

    def add_match(self, match_id: str, features: Dict[str, float], metadata: Dict[str, Any] = None):
        """Maçı vektörleştirir ve hafızaya ekler."""
        # Özellik sıralaması önemli! Her zaman aynı sırada olmalı.
        # Örnek: [home_xg, away_xg, home_shots, away_shots]
        # Şimdilik basitçe dict values alıyoruz ama production'da fixed schema şart.
        
        # Basit standardizasyon (vektör boyu 1 olsun diye değil, özellikler arası denge)
        # Ama şimdilik ham veri ile gidelim, cosine similarity yön bakar.
        
        # Örnek Feature Schema:
        # 1. Home xG
        # 2. Away xG
        # 3. Home Possession
        # 4. Away Possession
        
        vec_list = [
            features.get("home_xg", 0.0),
            features.get("away_xg", 0.0),
            features.get("home_possession", 50.0) / 100.0, # 0-1 arası
            features.get("away_possession", 50.0) / 100.0
        ]
        
        # Normale et (L2 vecor)
        vec_np = np.array(vec_list)
        norm = np.linalg.norm(vec_np)
        if norm > 0:
            vec_np = vec_np / norm
            
        self.vectors.append(MatchVector(match_id, vec_np.tolist(), metadata or {}))
        self.save()

    def find_similar(self, features: Dict[str, float], top_k: int = 3) -> List[Dict]:
        """Verilen özelliklere en benzer maçları bulur."""
        if not self.vectors:
            return []
            
        # Sorgu vektörü
        query_list = [
            features.get("home_xg", 0.0),
            features.get("away_xg", 0.0),
            features.get("home_possession", 50.0) / 100.0,
            features.get("away_possession", 50.0) / 100.0
        ]
        query_vec = np.array(query_list)
        q_norm = np.linalg.norm(query_vec)
        if q_norm > 0:
            query_vec = query_vec / q_norm
            
        results = []
        for mv in self.vectors:
            # Cosine Similarity = A . B (Normalize oldukları için)
            cand_vec = np.array(mv.vector)
            score = np.dot(query_vec, cand_vec)
            
            results.append({
                "match_id": mv.match_id,
                "score": float(score),
                "metadata": mv.metadata
            })
            
        # Skora göre sırala (Büyükten küçüğe)
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
