"""
ensemble.py – Quant Model Ensemble & Voting Mechanism.

This module aggregates predictions from multiple quantitative models
(Benter, Dixon-Coles, LSTM) to produce a robust consensus probability.
"""
from typing import Any, Dict, List
import numpy as np
from loguru import logger
from src.core.interfaces import QuantModel
from src.quant.adapters import BenterAdapter, LSTMAdapter, DixonColesAdapter

class EnsembleModel(QuantModel):
    """
    Ensemble Model aggregator.

    Logic:
    - Runs all registered sub-models.
    - Weights their predictions based on confidence (or static weights).
    - Calculates 'Consensus Score' (Variance of predictions).
    """

    def __init__(self):
        self.models: Dict[str, QuantModel] = {
            "benter": BenterAdapter(),
            "lstm": LSTMAdapter(),
            "dixon_coles": DixonColesAdapter()
        }
        # Static weights if confidence is unavailable
        self.weights = {
            "benter": 0.40,
            "dixon_coles": 0.40,
            "lstm": 0.20
        }

    def predict(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aggregates predictions.
        """
        results = {}
        probs_home = []
        probs_draw = []
        probs_away = []

        weighted_home = 0.0
        weighted_draw = 0.0
        weighted_away = 0.0
        total_weight = 0.0

        for name, model in self.models.items():
            try:
                res = model.predict(context)
                if "error" in res:
                    logger.warning(f"Model {name} failed: {res['error']}")
                    continue

                results[name] = res

                # Get probs
                ph = res.get("prob_home", 0.0)
                pd = res.get("prob_draw", 0.0)
                pa = res.get("prob_away", 0.0)

                # Normalize just in case
                s = ph + pd + pa
                if s > 0:
                    ph /= s
                    pd /= s
                    pa /= s

                probs_home.append(ph)
                probs_draw.append(pd)
                probs_away.append(pa)

                # Weighting
                w = self.weights.get(name, 0.33)
                # Boost weight by confidence if available
                conf = res.get("confidence", 0.5)
                # Simple boost: w * (0.5 + conf)
                final_w = w * (0.5 + conf)

                weighted_home += ph * final_w
                weighted_draw += pd * final_w
                weighted_away += pa * final_w
                total_weight += final_w

            except Exception as e:
                logger.error(f"Ensemble execution error ({name}): {e}")

        if total_weight == 0:
            # Fallback
            return {
                "prob_home": 0.33, "prob_draw": 0.33, "prob_away": 0.33,
                "confidence": 0.0,
                "consensus_score": 0.0,
                "details": results
            }

        final_home = weighted_home / total_weight
        final_draw = weighted_draw / total_weight
        final_away = weighted_away / total_weight

        # Calculate Consensus (Standard Deviation of Home Probs)
        # Low std dev = High Consensus
        if probs_home:
            consensus_std = float(np.std(probs_home))
            # Transform to score 0-1 (0=Chaos, 1=Agreement)
            # Max possible std for [0, 1] is 0.5. Usually around 0.1-0.2.
            # Score = 1 - (std / 0.25) clipped.
            consensus_score = max(0.0, 1.0 - (consensus_std * 4.0))
        else:
            consensus_score = 0.0

        return {
            "model": "ensemble",
            "prob_home": final_home,
            "prob_draw": final_draw,
            "prob_away": final_away,
            "confidence": max(final_home, final_draw, final_away), # Simple metric
            "consensus_score": consensus_score,
            "details": results
        }
