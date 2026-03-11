"""
lance_memory.py – LanceDB tabanlı çok-modlu semantik vektör veritabanı.
Metin, oran zaman serisi ve meta-veriyi vektör olarak saklar.
"""
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone

import numpy as np
from loguru import logger

LANCE_DIR = Path(__file__).resolve().parents[2] / "data" / "lancedb"


class LanceMemory:
    """Semantik vektör veritabanı – LanceDB."""

    def __init__(self, uri: str | Path = LANCE_DIR):
        self._uri = Path(uri)
        self._uri.mkdir(parents=True, exist_ok=True)
        self._db = None
        self._table = None
        self._init_db()

    def _init_db(self):
        try:
            import lancedb
            self._db = lancedb.connect(str(self._uri))
            if "embeddings" not in self._db.table_names():
                self._db.create_table(
                    "embeddings",
                    data=[{
                        "id": "init",
                        "text": "initialization",
                        "vector": np.zeros(384).tolist(),
                        "category": "system",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }],
                )
            self._table = self._db.open_table("embeddings")
            logger.debug("LanceMemory başlatıldı.")
        except ImportError:
            logger.warning("lancedb yüklü değil – LanceMemory devre dışı.")
        except Exception as e:
            logger.warning(f"LanceDB başlatma hatası: {e}")

    def _embed_text(self, text: str) -> list[float]:
        """Basit TF-IDF tabanlı embedding (ücretsiz, bağımlılıksız)."""
        tokens = text.lower().split()
        vec = np.zeros(384)
        for i, token in enumerate(tokens):
            idx = hash(token) % 384
            vec[idx] += 1.0 / (i + 1)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec.tolist()

    def add(self, doc_id: str, text: str, category: str = "general"):
        if self._table is None:
            return
        self._table.add([{
            "id": doc_id,
            "text": text,
            "vector": self._embed_text(text),
            "category": category,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }])

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        if self._table is None:
            return []
        q_vec = self._embed_text(query)
        try:
            results = self._table.search(q_vec).limit(top_k).to_list()
            return results
        except Exception as e:
            logger.warning(f"Lance arama hatası: {e}")
            return []

    def add_odds_event(self, match_id: str, market: str, odds: float, source: str):
        text = f"{match_id} {market} odds={odds:.2f} source={source}"
        self.add(doc_id=f"{match_id}_{market}_{odds}", text=text, category="odds")

    def add_news(self, headline: str, body: str, source: str):
        text = f"{headline} | {body} | {source}"
        doc_id = f"news_{hash(headline) % 999999}"
        self.add(doc_id=doc_id, text=text, category="news")
