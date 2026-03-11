"""
ensemble.py – Quant Model Ensemble & Voting Mechanism.

This module aggregates predictions from multiple quantitative models
(Benter, Dixon-Coles, LSTM, Bayesian, Quantum) to produce a robust consensus probability.
"""
from typing import Any, Dict, Optional
import time
from loguru import logger
from src.core.interfaces import QuantModel
from src.extensions.quantum_pricing_model import QuantumPricingModel
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
    - Detects 'Rotting Models' (Stale or drifting performance).
    """

    def __init__(self):
        self.bayesian = BayesianAdapter()
        self.quantum = QuantumAdapter()
        self.qpm = QuantumPricingModel()
        self.active_agent = container.get("active_agent")  # type: ignore

        # Initialize models
        self.models: Dict[str, QuantModel] = {
            "benter": BenterAdapter(),
            "lstm": LSTMAdapter(),
            "dixon_coles": DixonColesAdapter(),
            "bayesian": self.bayesian,
            "quantum": self.quantum,
            "qpm": self.qpm
        }

        # Static weights if confidence is unavailable
        self.weights = {
            "benter": 0.25,
            "dixon_coles": 0.20,
            "lstm": 0.15,
            "bayesian": 0.15,
            "quantum": 0.15,
            "qpm": 0.10
        }

        self.syndicate = SyndicateConsensus()

        # Model Health Monitoring
        self.model_health: Dict[str, Dict[str, Any]] = {
            name: {"last_success": time.time(), "errors": 0, "status": "HEALTHY"}
            for name in self.models
        }

    def _apply_chaos_veto(self, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        # --- 1. Physics Veto (Chaos Filter) ---
        chaos_regime = context.get("chaos_regime", "unknown")
        if chaos_regime == "chaotic":
            logger.warning(f"Ensemble: Chaos Veto activated for {context.get('match_id')}")
            return {
                "prob_home": 0.33, "prob_draw": 0.34, "prob_away": 0.33,
                "confidence": 0.0,
                "consensus_score": 0.0,
                "veto": "chaos",
                "details": {}
            }
        return None

    def _apply_dynamic_weights(self, context: Dict[str, Any]) -> None:
        # --- 2. Dynamic Weight Override (Strategy Evolver) ---
        if "ensemble_weights" in context:
            try:
                dynamic_weights = context["ensemble_weights"]
                for k, v in dynamic_weights.items():
                    if k in self.weights:
                        self.weights[k] = float(v)
            except Exception as e:
                logger.warning(f"Failed to apply dynamic weights: {e}")

    def _apply_active_inference_weights(self) -> None:
        # --- 2.5 Active Inference Weight Adjustment ---
        if self.active_agent:
            try:
                precision_weights = self.active_agent.get_precision_weights()
                for k, w in self.weights.items():
                    if k in precision_weights:
                        p_weight = precision_weights.get(k, precision_weights.get("default", 0.2))
                        self.weights[k] = (w * 0.7) + (p_weight * 0.3)
            except Exception as e:
                logger.warning(f"Failed to apply active inference weights: {e}")

    def _apply_quantum_boost(self, context: Dict[str, Any]) -> None:
        # --- 3. Quantum Boost ---
        quantum_conf = context.get("quantum_conf", 0.0)
        if quantum_conf > 0.8:
            self.weights["quantum"] = self.weights.get("quantum", 0.15) * 2.0
            logger.info(f"Ensemble: Quantum Boost activated (Conf: {quantum_conf:.2f})")

    def _determine_models_to_run(self, active_weights: Dict[str, float]) -> list:
        # Determine models to run
        models_to_run = []
        for name, model in self.models.items():
            if self.model_health[name]["status"] == "ROTTED" or self.model_health[name]["errors"] >= 3:
                logger.warning(f"Skipping rotted model: {name}")
                active_weights[name] = 0.0
                continue

            if active_weights.get(name, 0.0) < 0.05:
                logger.debug(f"Pruning model {name} dynamically due to low weight ({active_weights.get(name, 0.0):.3f})")
                self.model_health[name]["status"] = "PRUNED"
                active_weights[name] = 0.0
                continue

            self.model_health[name]["status"] = "HEALTHY"
            models_to_run.append((name, model))
        return models_to_run

    def _execute_models(self, models_to_run: list, context: Dict[str, Any]) -> Dict[str, Any]:
        results = {}
        for name, model in models_to_run:
            try:
                res = model.predict(context)
                if "error" in res:
                    logger.warning(f"Model {name} failed: {res['error']}")
                    self._record_failure(name)
                else:
                    self._record_success(name)
                    results[name] = res
            except Exception as e:
                logger.error(f"Ensemble execution error ({name}): {e}")
                self._record_failure(name)
        return results

    def _normalize_weights(self, results: Dict[str, Any], active_weights: Dict[str, float]) -> Dict[str, float]:
        # Normalize weights for active models
        total_weight = sum(active_weights[k] for k in results.keys() if k in active_weights)
        if total_weight > 0:
            return {k: active_weights[k] / total_weight for k in results.keys() if k in active_weights}
        return {k: 1.0 / len(results) for k in results.keys()}

    def predict(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aggregates predictions with Physics Veto and Quantum Boost.
        """
        veto = self._apply_chaos_veto(context)
        if veto:
            return veto

        self._apply_dynamic_weights(context)
        self._apply_active_inference_weights()
        self._apply_quantum_boost(context)

        active_weights = self.weights.copy()
        models_to_run = self._determine_models_to_run(active_weights)

        results = self._execute_models(models_to_run, context)

        if not results:
            return {
                "prob_home": 0.33, "prob_draw": 0.33, "prob_away": 0.33,
                "confidence": 0.0,
                "consensus_score": 0.0,
                "details": results
            }

        normalized_weights = self._normalize_weights(results, active_weights)

        # --- 4. Syndicate Adjudication ---
        verdict = self.syndicate.adjudicate(
            model_outputs=results,
            trust_scores=normalized_weights
        )

        if isinstance(verdict.audit_log, list):
             verdict.audit_log.append(f"Model Health: {str(self.model_health)}")
        elif isinstance(verdict.audit_log, dict):
             verdict.audit_log["model_health"] = self.model_health

        return {
            "model": "ensemble",
            "prob_home": verdict.prob_home,
            "prob_draw": verdict.prob_draw,
            "prob_away": verdict.prob_away,
            "confidence": verdict.confidence,
            "consensus_score": 1.0 - min(verdict.disagreement_level * 4.0, 1.0),
            "verdict_text": verdict.verdict_text,
            "syndicate_audit": verdict.audit_log,
            "details": results
        }

    def _record_success(self, model_name: str):
        """Updates health status on success."""
        self.model_health[model_name]["last_success"] = time.time()
        self.model_health[model_name]["errors"] = 0
        self.model_health[model_name]["status"] = "HEALTHY"

    def _record_failure(self, model_name: str):
        """Updates health status on failure."""
        self.model_health[model_name]["errors"] += 1
        if self.model_health[model_name]["errors"] >= 3:
            self.model_health[model_name]["status"] = "ROTTED"
            logger.error(f"Model {model_name} marked as ROTTED due to repeated failures.")

    def reset_health(self, model_name: Optional[str] = None):
        """Manually resets health status (e.g. after fix/redeploy)."""
        if model_name:
             self.model_health[model_name] = {"last_success": time.time(), "errors": 0, "status": "HEALTHY"}
             logger.info(f"Health reset for {model_name}")
        else:
            for name in self.models:
                self.model_health[name] = {"last_success": time.time(), "errors": 0, "status": "HEALTHY"}
            logger.info("Health reset for all models.")
