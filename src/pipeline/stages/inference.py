from typing import Any, Dict, List
import asyncio
from loguru import logger
import polars as pl

from src.pipeline.core import PipelineStage
from src.system.container import container
from src.ingestion.news_rag import NewsRAGAnalyzer

class InferenceStage(PipelineStage):
    """
    Quant Modelleri ve AI Analiz Motoru.

    Görevleri:
    1. İstatistiksel Modelleri Çalıştır (Poisson, Elo, Glicko)
    2. Deep Learning Modellerini Çalıştır (LSTM, Transformer)
    3. Haber/Sentiment Analizi (RAG)
    4. Tüm sinyalleri ham haliyle Ensemble katmanına ilet.
    """

    def __init__(self):
        super().__init__("inference")
        self.prob_engine = container.get("prob_engine")

        # RAG Analizcisi (Token varsa çalışır)
        try:
            self.rag = NewsRAGAnalyzer()
        except Exception as e:
            logger.warning(f"NewsRAG başlatılamadı: {e}")
            self.rag = None

        # Diğer modeller (Lazy loading)
        self.models = {}
        self._load_models()

    def _load_models(self):
        """Quant modellerini yükle."""
        try:
            from src.quant.lstm_trend import LSTMTrendAnalyzer
            self.models["lstm"] = LSTMTrendAnalyzer(hidden_size=32, num_layers=2)
            # self.models["lstm"].load_model() # Eğer kayıtlı model varsa
        except ImportError:
            pass

        try:
            from src.quant.benter_model import BenterModel
            self.benter_model = BenterModel()
        except ImportError:
            self.benter_model = None
            logger.warning("BenterModel yüklenemedi.")

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Tüm maçlar için paralel analiz yap."""
        matches = context.get("matches", pl.DataFrame())
        features = context.get("features", pl.DataFrame())

        if matches.is_empty():
            logger.info("Analiz edilecek maç yok.")
            return {"match_predictions": {}}

        match_predictions = {}
        tasks = []

        # Her maç için bir analiz görevi oluştur
        for row in matches.iter_rows(named=True):
            tasks.append(self._analyze_single_match(row, features))

        # Paralel çalıştır
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for res in results:
            if isinstance(res, dict) and "match_id" in res:
                match_predictions[res["match_id"]] = res
            elif isinstance(res, Exception):
                logger.error(f"Maç analizi hatası: {res}")

        return {"match_predictions": match_predictions}

    async def _analyze_single_match(self, match: Dict[str, Any], features: pl.DataFrame) -> Dict[str, Any]:
        """Tek bir maçı tüm boyutlarıyla analiz et."""
        home = match.get("home_team")
        away = match.get("away_team")
        match_id = f"{home}_{away}"

        signals = {
            "match_id": match_id,
            "home_team": home,
            "away_team": away,
            "prob_engine": 0.0,
            "lstm": 0.0,
            "news_sentiment": 0.5,
            "news_summary": ""
        }

        # 1. Probabilistic Engine (Bill Benter Logic)
        if hasattr(self, "benter_model") and self.benter_model:
            try:
                # Feature'lardan xG çekmeye çalış, yoksa default 1.35
                # İleride: features.filter(pl.col("match_id") == match_id)
                home_xg = 1.35
                away_xg = 1.10

                # Context (Hava, Sakatlık) - Gelecekte feature'dan gelecek
                ctx = {"rain": False}

                probs = self.benter_model.calculate_benter_probabilities(home_xg, away_xg, ctx)
                signals["prob_engine"] = probs["prob_home"]
                signals["benter_probs"] = probs # Detaylı veri
            except Exception as e:
                logger.warning(f"BenterModel hatası {match_id}: {e}")
        elif self.prob_engine:
            # Fallback
            try:
                signals["prob_engine"] = 0.55
            except Exception:
                pass

        # 2. LSTM / Deep Learning (CPU/GPU Bound)
        if "lstm" in self.models:
            # await asyncio.to_thread(self.models["lstm"].predict, ...)
            signals["lstm"] = 0.60 # Mock

        # 3. News RAG (I/O Bound - API Call)
        if self.rag:
            try:
                # Takımların son durumunu analiz et
                # API kotasını korumak için sadece önemli maçlarda çalıştırılabilir
                # Şimdilik her maçta deneyelim (async olduğu için hızlı)
                rag_res = await self.rag.analyze_match(home, away)
                signals["news_sentiment"] = rag_res.get("sentiment_diff", 0.0) + 0.5 # -1..1 -> 0..1 scale (kabaca)
                # Düzeltme: sentiment_diff zaten fark.
                # ensemble logic: <0.4 negatif, >0.6 pozitif.
                # Eğer home sentiment 0.8, away 0.4 ise diff +0.4.
                # Bunu 0.5 merkezli bir skora çevirelim: 0.5 + (diff / 2)
                # Max diff 1.0 -> 1.0 score. Min diff -1.0 -> 0.0 score.
                diff = rag_res.get("sentiment_diff", 0.0)
                signals["news_sentiment"] = 0.5 + (diff / 2)

                signals["news_summary"] = (
                    f"Home: {rag_res.get('home_summary')[:50]}... | "
                    f"Away: {rag_res.get('away_summary')[:50]}..."
                )
            except Exception as e:
                logger.warning(f"RAG hatası {match_id}: {e}")

        return signals
