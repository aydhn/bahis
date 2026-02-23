"""
ensemble.py – The Board of Directors (Karar Mekanizması).

Bu modül, farklı modellerden (Quant, ML, AI, RAG) gelen tahminleri toplar
ve "Ortak Akıl" (Ensemble) ile nihai kararı verir.

Mantık:
  - Voting Regressor / Classifier mantığı.
  - Haber/Sentiment (RAG) sinyallerini "veto" veya "boost" olarak kullanır.
  - Modellerin tarihsel başarısına göre ağırlıklandırma (ileride).
"""
from typing import Any, Dict, List

import numpy as np
from loguru import logger

from src.pipeline.core import PipelineStage


class EnsembleStage(PipelineStage):
    """Farklı model çıktılarını birleştiren 'Beyin' katmanı."""

    def __init__(self):
        super().__init__("ensemble")
        # Model güven ağırlıkları (Zamanla optimize edilebilir)
        self.weights = {
            "prob_engine": 0.4,
            "lstm": 0.3,
            "poisson": 0.2,
            "xgboost": 0.1,  # Eğer varsa
        }

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Tüm tahminleri al, birleştir ve nihai olasılığı üret."""
        raw_preds = context.get("predictions", {})
        matches = context.get("matches")

        if not raw_preds:
            logger.debug("Ensemble: Tahmin yok.")
            return {"ensemble_results": []}

        final_decisions = []

        # Olay: Her maç için tüm modellerin ne dediğine bak
        # raw_preds yapısı: {"model_name": [ {match_id: ..., prob_home: ...}, ... ]}
        # Bunu maç bazlı bir yapıya çevirmemiz lazım.
        # Varsayım: InferenceStage, her model için sıralı liste dönüyor ve matches ile aynı sırada.

        # Bu biraz kırılgan. InferenceStage'i refactor ederken her maç için bir dict listesi döndürmesini sağlayacağım.
        # Şimdilik context'ten "match_predictions" bekleyelim.
        # match_predictions = { "match_id": { "model_A": prob, "model_B": prob, "news_sentiment": ... } }

        match_preds = context.get("match_predictions", {})

        for match_id, signals in match_preds.items():
            home_team = signals.get("home_team", "Unknown")
            away_team = signals.get("away_team", "Unknown")

            # 1. Ağırlıklı Ortalama
            weighted_sum = 0.0
            total_weight = 0.0

            model_votes = []

            for model_name, weight in self.weights.items():
                if model_name in signals:
                    prob = signals[model_name]
                    # Prob bazen dict olabilir {home: 0.5, draw: 0.3...}
                    # Şimdilik sadece home win prob alalım
                    p_home = prob if isinstance(prob, (float, int)) else prob.get("prob_home", 0.0)

                    weighted_sum += p_home * weight
                    total_weight += weight
                    model_votes.append(f"{model_name}={p_home:.2f}")

            if total_weight == 0:
                continue

            base_prob = weighted_sum / total_weight

            # 2. Sentiment Ayarı (RAG)
            # Sentiment 0-1 arası. 0.5 nötr.
            # <0.4 ise negatif etki, >0.6 ise pozitif etki.
            sentiment_score = signals.get("news_sentiment", 0.5)
            sentiment_impact = 0.0

            if sentiment_score < 0.4:
                sentiment_impact = -0.10 * (0.4 - sentiment_score) * 10 # Örn: 0.2 -> -0.20
            elif sentiment_score > 0.6:
                sentiment_impact = 0.05 * (sentiment_score - 0.6) * 10

            final_prob = base_prob + sentiment_impact
            final_prob = max(0.0, min(1.0, final_prob))

            # 3. Güven Skoru (Modeller arası uyum)
            # Standart sapma düşükse güven yüksek
            probs = [float(x.split('=')[1]) for x in model_votes]
            std_dev = np.std(probs) if probs else 0.0
            confidence = 1.0 - std_dev  # Basit bir yaklaşım

            decision = {
                "match_id": match_id,
                "home_team": home_team,
                "away_team": away_team,
                "prob_home": final_prob,
                "raw_prob": base_prob,
                "sentiment_adj": sentiment_impact,
                "confidence": confidence,
                "votes": model_votes,
                "news_summary": signals.get("news_summary", "")
            }
            final_decisions.append(decision)

            logger.info(
                f"Ensemble {match_id}: Base={base_prob:.2f} + Sent={sentiment_impact:.2f} "
                f"-> Final={final_prob:.2f} (Conf: {confidence:.2f})"
            )

        return {"ensemble_results": final_decisions}
