"""
ensemble.py – Quant Model Ensemble & Voting Mechanism.

This module aggregates predictions from multiple quantitative models
(Benter, Dixon-Coles, LSTM, Bayesian, Quantum) to produce a robust consensus probability.
"""
from typing import Any, Dict, List
import numpy as np
from loguru import logger
from src.core.interfaces import QuantModel
from src.quant.adapters import BenterAdapter, LSTMAdapter, DixonColesAdapter, BayesianAdapter, QuantumAdapter
from src.system.container import container
from src.quant.analysis.syndicate_consensus import SyndicateConsensus

class EnsembleModel(QuantModel):
    """
    Ensemble Model aggregator.

    Logic:
    - Runs all registered sub-models.
    - Weights their predictions based on confidence (or static weights).
    - Calculates 'Consensus Score' (Variance of predictions).
    """

    def __init__(self):
        self.bayesian = BayesianAdapter()
        self.quantum = QuantumAdapter()
        self.active_agent = container.get("active_agent")
        self.models: Dict[str, QuantModel] = {
            "benter": BenterAdapter(),
            "lstm": LSTMAdapter(),
            "dixon_coles": DixonColesAdapter(),
            "bayesian": self.bayesian,
            "quantum": self.quantum
        }
        # Static weights if confidence is unavailable
        self.weights = {
            "benter": 0.30,
            "dixon_coles": 0.25,
            "lstm": 0.15,
            "bayesian": 0.15,
            "quantum": 0.15
        }
        self.syndicate = SyndicateConsensus()

    def predict(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aggregates predictions with Physics Veto and Quantum Boost.
        """
        # --- 1. Physics Veto (Chaos Filter) ---
        # If chaos is detected, we might reject the bet immediately or switch to contrarian mode.
        chaos_regime = context.get("chaos_regime", "unknown")

        if chaos_regime == "chaotic":
            # If chaotic, return neutral/uncertain result
            logger.warning(f"Ensemble: Chaos Veto activated for {context.get('match_id')}")
            return {
                "prob_home": 0.33, "prob_draw": 0.34, "prob_away": 0.33,
                "confidence": 0.0,
                "consensus_score": 0.0,
                "veto": "chaos",
                "details": {}
            }

        # --- 2. Dynamic Weight Override (Strategy Evolver) ---
        if "ensemble_weights" in context:
            try:
                dynamic_weights = context["ensemble_weights"]
                # Normalize keys just in case
                for k, v in dynamic_weights.items():
                    if k in self.weights:
                        self.weights[k] = float(v)
            except Exception as e:
                logger.warning(f"Failed to apply dynamic weights: {e}")

        # --- 2.5 Active Inference Weight Adjustment ---
        # Adjust weights based on precision (trust) from ActiveInferenceAgent
        if self.active_agent:
            try:
                precision_weights = self.active_agent.get_precision_weights()
                # Blend static/evolved weights with active precision weights
                # Formula: new_weight = (static * 0.7) + (precision * 0.3)
                for k, w in self.weights.items():
                    if k in precision_weights:
                        # Map model names if necessary (e.g. benter -> poisson in agent)
                        # Assuming 1-to-1 mapping or close enough
                        p_weight = precision_weights.get(k, precision_weights.get("default", 0.2))
                        self.weights[k] = (w * 0.7) + (p_weight * 0.3)
                        # logger.debug(f"Ensemble: Adjusted {k} weight to {self.weights[k]:.3f} via Active Inference")
            except Exception as e:
                logger.warning(f"Failed to apply active inference weights: {e}")

        # --- 3. Quantum Boost ---
        # If Quantum Brain is highly confident, boost its weight
        quantum_conf = context.get("quantum_conf", 0.0)
        if quantum_conf > 0.8:
            self.weights["quantum"] = self.weights.get("quantum", 0.15) * 2.0
            logger.info(f"Ensemble: Quantum Boost activated (Conf: {quantum_conf:.2f})")

        results = {}
        for name, model in self.models.items():
            try:
                res = model.predict(context)
                if "error" in res:
                    logger.warning(f"Model {name} failed: {res['error']}")
                    continue
                results[name] = res
            except Exception as e:
                logger.error(f"Ensemble execution error ({name}): {e}")

        if not results:
            return {
                "prob_home": 0.33, "prob_draw": 0.33, "prob_away": 0.33,
                "confidence": 0.0,
                "consensus_score": 0.0,
                "details": results
            }

        # --- 4. Syndicate Adjudication ---
        # Delegate weighting and consensus logic to Syndicate
        verdict = self.syndicate.adjudicate(
            model_outputs=results,
            trust_scores=self.weights
        )

        return {
            "model": "ensemble",
            "prob_home": verdict.prob_home,
            "prob_draw": verdict.prob_draw,
            "prob_away": verdict.prob_away,
            "confidence": verdict.confidence,
            "consensus_score": 1.0 - min(verdict.disagreement_level * 4.0, 1.0), # Map 0.25 std to 0 score
            "verdict_text": verdict.verdict_text,
            "syndicate_audit": verdict.audit_log,
            "details": results
        }
