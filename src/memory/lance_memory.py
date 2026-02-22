"""
lance_memory.py – LanceDB tabanlı çok-modlu semantik vektör veritabanı.

Metin, oran zaman serisi ve meta-veriyi vektör olarak saklar.
Sentence-Transformer olmadan çalışan ücretsiz hash-based embedding ile
semantik benzerlik araması yapar.

Kavramlar:
  - Vector Database: Vektör uzayında benzerlik araması
  - Embedding: Metni sayısal vektöre dönüştürme
  - Hash Embedding: Bağımlılıksız, hızlı, deterministik embedding
  - Cosine Similarity: Vektörler arası açısal benzerlik ölçüsü
  - HNSW Index: Approximate Nearest Neighbor arama yapısı

Akış:
  1. Metin girdi → hash-based embedding (384-dim)
  2. LanceDB tablosuna vektör + metadata olarak ekle
  3. Sorgu → vektörleştir → en yakın K sonucu getir
  4. Sonuçları sırala ve döndür
"""
from __future__ import annotations

import math
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

import numpy as np
from loguru import logger

LANCE_DIR = Path(__file__).resolve().parents[2] / "data" / "lancedb"


def _safe_float(value, default: float = 0.0) -> float:
    """None, NaN ve geçersiz değerleri güvenli float'a çevirir."""
    if value is None:
        return default
    try:
        f = float(value)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def _safe_str(value, default: str = "") -> str:
    """None ve geçersiz değerleri güvenli str'ye çevirir."""
    if value is None:
        return default
    return str(value)


class LanceMemory:
    """Semantik vektör veritabanı – LanceDB.

    Kullanım:
        lance = LanceMemory()
        lance.add("doc1", "Galatasaray maçında sürpriz gol", category="news")
        results = lance.search("sürpriz gol")
    """

    EMBED_DIM = 384

    def __init__(self, uri: str | Path = LANCE_DIR):
        self._uri = Path(uri)
        self._uri.mkdir(parents=True, exist_ok=True)
        self._db: Any = None
        self._table: Any = None
        self._add_count = 0
        self._search_count = 0
        self._init_db()

    def _init_db(self):
        try:
            import lancedb
            self._db = lancedb.connect(str(self._uri))
            _tables = self._db.list_tables() if hasattr(self._db, "list_tables") else self._db.table_names()
            if "embeddings" not in _tables:
                try:
                    self._db.create_table(
                        "embeddings",
                        data=[{
                            "id": "init",
                            "text": "initialization",
                            "vector": np.zeros(self.EMBED_DIM).tolist(),
                            "category": "system",
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }],
                    )
                except Exception as create_err:
                    if "already exists" not in str(create_err).lower():
                        raise
                    logger.debug("LanceDB embeddings tablosu zaten mevcut – açılıyor.")
            self._table = self._db.open_table("embeddings")
            logger.debug("LanceMemory başlatıldı.")
        except ImportError:
            logger.warning("lancedb yüklü değil – LanceMemory devre dışı.")
        except Exception as e:
            logger.warning(f"LanceDB başlatma hatası: {e}")

    def _embed_text(self, text: str) -> list[float]:
        """Hash-based character n-gram embedding (ücretsiz, deterministik).

        Bigram + trigram hash'leri kullanarak daha zengin temsil üretir.
        """
        vec = np.zeros(self.EMBED_DIM)
        text_lower = text.lower().strip()
        if not text_lower:
            return vec.tolist()

        tokens = text_lower.split()
        for i, token in enumerate(tokens):
            weight = 1.0 / (i + 1)
            idx = hash(token) % self.EMBED_DIM
            vec[idx] += weight

            for ng_len in (2, 3):
                for j in range(max(0, len(token) - ng_len + 1)):
                    ngram = token[j:j + ng_len]
                    ng_idx = hash(ngram) % self.EMBED_DIM
                    vec[ng_idx] += weight * 0.3

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec.tolist()

    def add(self, doc_id: str, text: str, category: str = "general"):
        """Vektör veritabanına doküman ekle."""
        if self._table is None:
            return
        try:
            self._table.add([{
                "id": _safe_str(doc_id, "unknown"),
                "text": _safe_str(text, ""),
                "vector": self._embed_text(_safe_str(text, "")),
                "category": _safe_str(category, "general"),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }])
            self._add_count += 1
        except Exception as e:
            logger.debug(f"[Lance] Ekleme hatası: {e}")

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Semantik benzerlik araması."""
        if self._table is None:
            return []
        self._search_count += 1
        q_vec = self._embed_text(query)
        try:
            results = self._table.search(q_vec).limit(top_k).to_list()
            return results
        except Exception as e:
            logger.warning(f"Lance arama hatası: {e}")
            return []

    def find_similar_matches(self, match_id: str, limit: int = 5) -> list[dict]:
        """Belirli bir maça benzeyen geçmiş maçları bulur."""
        # Maçın kendi metin temsilini bul
        try:
            res = self._table.search().where(f"id LIKE '%{match_id}%'").limit(1).to_list()
            if not res:
                return []
            return self.search(res[0]["text"], top_k=limit+1)[1:] # Kendisi hariç
        except Exception:
            return []

    def add_odds_event(self, match_id: str, market: str,
                       odds: Any, source: str):
        """Oran olayını semantik hafızaya kaydet."""
        match_id = _safe_str(match_id, "unknown")
        market = _safe_str(market, "1X2")
        source = _safe_str(source, "unknown")
        odds_f = _safe_float(odds, 0.0)
        text = f"{match_id} {market} odds={odds_f:.2f} source={source}"
        self.add(
            doc_id=f"{match_id}_{market}_{odds_f:.2f}",
            text=text,
            category="odds",
        )

    def add_news(self, headline: str, body: str, source: str):
        """Haber haberini semantik hafızaya kaydet."""
        headline = _safe_str(headline, "")
        body = _safe_str(body, "")
        source = _safe_str(source, "")
        text = f"{headline} | {body} | {source}"
        doc_id = f"news_{hash(headline) % 999999}"
        self.add(doc_id=doc_id, text=text, category="news")

    def add_analysis(self, match_id: str, analysis_text: str,
                     analysis_type: str = "general"):
        """Analiz sonucunu semantik hafızaya kaydet."""
        self.add(
            doc_id=f"analysis_{match_id}_{analysis_type}",
            text=f"{match_id} [{analysis_type}]: {analysis_text}",
            category="analysis",
        )

    def get_stats(self) -> dict:
        """Hafıza istatistikleri."""
        return {
            "add_count": self._add_count,
            "search_count": self._search_count,
            "table_ready": self._table is not None,
        }
