from typing import Dict, Any
from loguru import logger
import polars as pl
from src.pipeline.core import PipelineStage
from src.system.container import container

class InferenceStage(PipelineStage):
    """Generates predictions using various ML/Quant models."""

    def __init__(self):
        super().__init__("inference")
        self.prob_engine = container.get("prob_engine")
        self.graph_rag = container.get("graph_rag")

        # Additional models (mocked or loaded if available)
        try:
            from src.quant.lstm_trend import LSTMTrendAnalyzer
            self.lstm = LSTMTrendAnalyzer(hidden_size=32, num_layers=2)
            self.lstm.load_model()
        except ImportError:
            self.lstm = None

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        features = context.get("features", pl.DataFrame())
        matches = context.get("matches", pl.DataFrame())

        if features.is_empty():
            return {"predictions": {}}

        preds = {}

        # 1. Probabilistic Engine
        if self.prob_engine:
             preds["prob_engine"] = self.prob_engine.predict(features)

        # 2. LSTM Trend Analysis (requires match details)
        if self.lstm:
            lstm_results = []
            for row in matches.iter_rows(named=True):
                home, away = row.get("home_team"), row.get("away_team")
                # Need history from DB (mock for now as DB isn't injected here)
                # In real flow, we'd fetch history
                res = self.lstm.predict_for_match(home, away, [], [])
                lstm_results.append(res)
            preds["lstm"] = lstm_results

        # 3. GraphRAG Crisis Analysis
        if self.graph_rag:
            rag_results = []
            for row in matches.iter_rows(named=True):
                try:
                    home, away = row.get("home_team"), row.get("away_team")
                    crisis = self.graph_rag.analyze_crisis(home)
                    rag_results.append({
                        "team": home,
                        "score": crisis.crisis_score,
                        "level": crisis.crisis_level
                    })
                except Exception as e:
                    logger.warning(f"GraphRAG failed for {home}: {e}")
            preds["graph_rag"] = rag_results

        # 4. Ensemble (Simple average or stacking)
        # For simplicity, we just pass the prob_engine results as 'final'
        # In a real pipeline, we'd have an EnsembleStage
        final_probs = preds.get("prob_engine", [])

        return {"predictions": preds, "ensemble": final_probs}
