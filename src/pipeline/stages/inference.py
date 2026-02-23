from typing import Any, Dict, List
import asyncio
from loguru import logger
import polars as pl

from src.pipeline.core import PipelineStage
from src.system.container import container
from src.core.model_registry import ModelRegistry
from src.ingestion.news_rag import NewsRAGAnalyzer

class InferenceStage(PipelineStage):
    """
    Quant Modelleri ve AI Analiz Motoru.
    Dinamik model yükleme (ModelRegistry) ile çalışır.
    """

    def __init__(self):
        super().__init__("inference")

        # Modelleri Kaydet
        ModelRegistry.load_defaults()
        self.models = ModelRegistry.get_all_models()

        # RAG Analizcisi (Opsiyonel)
        try:
            self.rag = NewsRAGAnalyzer()
        except Exception:
            self.rag = None

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Tüm maçlar için paralel analiz yap."""
        matches = context.get("matches", pl.DataFrame())
        features = context.get("features", pl.DataFrame())

        if matches.is_empty():
            logger.info("Analiz edilecek maç yok.")
            return {"ensemble_results": []}

        tasks = []
        # Her maç için feature satırını bul ve analiz et
        # Optimize: features DataFrame'ini hash map'e çevir (match_id -> row)
        feat_map = {row["match_id"]: row for row in features.iter_rows(named=True)}

        for row in matches.iter_rows(named=True):
            match_feat = feat_map.get(row["match_id"], {})
            # Match ve Feature verilerini birleştir
            full_context = {**row, **match_feat}
            tasks.append(self._analyze_single_match(full_context))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        valid_results = []
        for res in results:
            if isinstance(res, dict):
                valid_results.append(res)
            elif isinstance(res, Exception):
                logger.error(f"Maç analizi hatası: {res}")

        return {"ensemble_results": valid_results}

    async def _analyze_single_match(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Tek bir maçı tüm modellerle analiz et."""
        match_id = context.get("match_id", "Unknown")

        predictions = {
            "match_id": match_id,
            "home_team": context.get("home_team"),
            "away_team": context.get("away_team"),
            "prob_home": 0.0,
            "details": {}
        }

        # 1. Quant Modellerini Çalıştır (Parallel execution in thread pool)
        # CPU-bound işlemleri thread'e atıyoruz
        model_tasks = [
            asyncio.to_thread(model.predict, context)
            for model in self.models
        ]

        model_results = await asyncio.gather(*model_tasks, return_exceptions=True)

        total_prob = 0.0
        count = 0

        for res in model_results:
            if isinstance(res, dict) and "prob_home" in res:
                model_name = res.get("model", "unknown")
                predictions["details"][model_name] = res

                # Basit ortalama (Ensemble logic buraya daha sonra eklenebilir)
                prob = res.get("prob_home", 0.5)
                conf = res.get("confidence", 0.5)

                # Ağırlıklı ortalama
                weight = conf if conf > 0 else 0.5
                total_prob += prob * weight
                count += weight

        if count > 0:
            predictions["prob_home"] = total_prob / count
        else:
            predictions["prob_home"] = 0.5  # Fallback

        # 2. News RAG (I/O Bound)
        if self.rag:
            try:
                rag_res = await self.rag.analyze_match(
                    context.get("home_team"),
                    context.get("away_team")
                )
                predictions["news_summary"] = rag_res.get("summary", "")
                predictions["news_sentiment"] = rag_res.get("sentiment_score", 0.0)
            except Exception:
                pass

        return predictions
