"""
syndicate_consensus.py – The "Syndicate" Voting Mechanism.

Mimics a team of expert gamblers (Bill Benter style) debating a match.
If models disagree significantly, a "Debate" is triggered, and
the final decision is weighted by each model's recent "Trust Score".
"""
import numpy as np
from loguru import logger
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass

@dataclass
class ConsensusVerdict:
    """The final decision of the Syndicate."""
    prob_home: float
    prob_draw: float
    prob_away: float
    confidence: float
    disagreement_level: float # 0.0 (Unified) -> 1.0 (Civil War)
    verdict_text: str
    audit_log: List[str]

class SyndicateConsensus:
    """
    Arbitrates between conflicting model predictions.
    """

    def adjudicate(self, model_outputs: Dict[str, Dict[str, float]],
                   trust_scores: Dict[str, float] = None) -> ConsensusVerdict:
        """
        Synthesizes a consensus from multiple model outputs.

        Args:
            model_outputs: { 'benter': {'prob_home': 0.6, ...}, 'lstm': ... }
            trust_scores: Optional manual trust weights (default: equal or 1.0)
        """
        if not model_outputs:
            return ConsensusVerdict(0.33, 0.33, 0.33, 0.0, 0.0, "No models consulted.", [])

        # 1. Extract Vectors
        home_probs = []
        draw_probs = []
        away_probs = []
        weights = []
        names = []

        audit_log = []

        for name, pred in model_outputs.items():
            if "prob_home" not in pred: continue

            p_h = pred.get("prob_home", 0.33)
            p_d = pred.get("prob_draw", 0.33)
            p_a = pred.get("prob_away", 0.33)

            # Trust Score (Default to 1.0 if not provided)
            # Active Inference Agent provides these based on recent accuracy
            trust = trust_scores.get(name, 1.0) if trust_scores else 1.0

            # Boost trust if model is very confident?
            # confidence = pred.get("confidence", 0.5)
            # trust *= (0.5 + confidence)

            home_probs.append(p_h)
            draw_probs.append(p_d)
            away_probs.append(p_a)
            weights.append(trust)
            names.append(name)

            audit_log.append(f"{name}: Home={p_h:.2f}, Trust={trust:.2f}")

        if not home_probs:
            return ConsensusVerdict(0.33, 0.33, 0.33, 0.0, 0.0, "No valid predictions.", [])

        # 2. Calculate Statistics
        h_arr = np.array(home_probs)
        w_arr = np.array(weights)

        # Weighted Average
        if w_arr.sum() == 0: w_arr = np.ones_like(w_arr)

        final_h = np.average(h_arr, weights=w_arr)
        final_d = np.average(draw_probs, weights=w_arr)
        final_a = np.average(away_probs, weights=w_arr)

        # Disagreement (Std Dev of Home Prob)
        disagreement = np.std(h_arr)

        # 3. Formulate Verdict
        verdict = "Consensus Reached."
        if disagreement > 0.15:
            # Find the dissenters
            # Who is furthest from the mean?
            diffs = np.abs(h_arr - final_h)
            dissenter_idx = np.argmax(diffs)
            dissenter_name = names[dissenter_idx]
            verdict = f"High Conflict! {dissenter_name} disagrees with the group."

            # Reduce confidence if disagreement is high
            # We don't change probabilities, but we flag it

        elif disagreement < 0.05:
            verdict = "Strong Unanimous Consensus."

        # Confidence Penalty based on Disagreement
        # Base confidence is the max prob
        raw_conf = max(final_h, final_d, final_a)

        # Penalty: If models fight, we are less sure.
        # e.g. std=0.2 -> penalty=0.1
        conf_penalty = disagreement * 0.5
        final_conf = max(0.0, raw_conf - conf_penalty)

        return ConsensusVerdict(
            prob_home=float(final_h),
            prob_draw=float(final_d),
            prob_away=float(final_a),
            confidence=float(final_conf),
            disagreement_level=float(disagreement),
            verdict_text=verdict,
            audit_log=audit_log
        )
