from typing import Any, Dict, List
import asyncio
from loguru import logger
import polars as pl

from src.pipeline.core import PipelineStage
from src.system.container import container
from src.core.model_registry import ModelRegistry
from src.ingestion.news_rag import NewsRAGAnalyzer
from src.quant.models.ensemble import EnsembleModel
from src.quant.risk.entropy_kelly import EntropyKelly
from src.quant.analysis.similarity import SimilarityEngine
from src.quant.meta_labeling import MetaLabeler
import numpy as np

class InferenceStage(PipelineStage):
    """
    Quant Models & AI Analysis Engine.
    Uses EnsembleModel for robust predictions and EntropyKelly for uncertainty metrics.
    Integrated with SimilarityEngine (Pattern Matching) and MetaLabeler (Error Correction).
    """

    def __init__(self):
        super().__init__("inference")

        # Initialize Ensemble Model (Aggregates Benter, Dixon-Coles, LSTM)
        self.ensemble = EnsembleModel()

        # Initialize Entropy Calculator
        self.entropy_calc = EntropyKelly()

        # Initialize Advanced Quant Engines
        self.similarity_engine = SimilarityEngine()
        self.similarity_engine.mock_fit() # TODO: Fit on real DB history

        self.meta_labeler = MetaLabeler()
        self.meta_labeler.mock_train() # TODO: Train on real history

        # RAG Analyzer (Optional)
        try:
            self.rag = NewsRAGAnalyzer()
        except Exception:
            self.rag = None

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Parallel analysis for all matches."""
        matches = context.get("matches", pl.DataFrame())

        # Merge mocked features if available (Autonomous Mode)
        features = context.get("features", pl.DataFrame())
        mock_features = context.get("mock_features")

        if mock_features is not None:
            if features.is_empty():
                features = mock_features
                logger.info("Using Mock Features for Inference.")
            else:
                # Append? Or override? usually Mock is fallback.
                pass

        if matches.is_empty():
            logger.info("No matches to analyze.")
            return {"ensemble_results": []}

        tasks = []
        # Optimize: Create feature map
        feat_map = {row["match_id"]: row for row in features.iter_rows(named=True)}

        for row in matches.iter_rows(named=True):
            match_feat = feat_map.get(row["match_id"], {})
            full_context = {**row, **match_feat}
            tasks.append(self._analyze_single_match(full_context))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        valid_results = []
        for res in results:
            if isinstance(res, dict):
                valid_results.append(res)
            elif isinstance(res, Exception):
                logger.error(f"Match Analysis Error: {res}")

        return {"ensemble_results": valid_results}

    async def _analyze_single_match(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a single match using Ensemble, RAG, Similarity and Meta-Labeling."""
        match_id = context.get("match_id", "Unknown")

        # 1. Ensemble Prediction (CPU Bound -> Thread)
        prediction = await asyncio.to_thread(self.ensemble.predict, context)

        # 2. Entropy Calculation
        probs = [
            prediction.get("prob_home", 0.0),
            prediction.get("prob_draw", 0.0),
            prediction.get("prob_away", 0.0)
        ]
        entropy = self.entropy_calc.calculate_entropy(probs)
        prediction["entropy"] = entropy

        # 3. News RAG (I/O Bound)
        if self.rag:
            try:
                rag_res = await self.rag.analyze_match(
                    context.get("home_team"),
                    context.get("away_team")
                )
                prediction["news_summary"] = rag_res.get("summary", "")
                prediction["news_sentiment"] = rag_res.get("sentiment_score", 0.0)
            except Exception:
                pass

        # 4. Pattern Matching (Similarity Engine)
        try:
            # Construct a feature vector from prediction data for now
            # In production, this should be pre-match technical features
            feat_vec = np.array([[
                prediction.get("prob_home", 0.33),
                prediction.get("prob_draw", 0.33),
                prediction.get("prob_away", 0.33),
                prediction.get("home_xg", 1.0),
                prediction.get("away_xg", 1.0)
            ]])
            # Pad if needed to match mock_fit shape (5 features)
            if feat_vec.shape[1] < 5:
                feat_vec = np.pad(feat_vec, ((0,0), (0, 5-feat_vec.shape[1])))

            similar_matches = self.similarity_engine.find_similar(feat_vec[:, :5])
            prediction["similar_matches"] = similar_matches
        except Exception as e:
            logger.warning(f"Similarity search failed: {e}")

        # 5. Meta-Labeling (Quality Check)
        try:
            # We need Odds to assess quality properly. Assuming standard 2.0 if missing.
            meta_features = {
                "confidence": max(probs),
                "entropy": entropy,
                "odds": context.get("odds_home", 2.0) # Simplified
            }
            quality_score = self.meta_labeler.predict_score(meta_features)
            prediction["meta_quality_score"] = quality_score
        except Exception as e:
            logger.warning(f"Meta-labeling failed: {e}")
            prediction["meta_quality_score"] = 0.5

        # Add basic identification
        prediction["match_id"] = match_id
        prediction["home_team"] = context.get("home_team")
        prediction["away_team"] = context.get("away_team")

        return prediction
