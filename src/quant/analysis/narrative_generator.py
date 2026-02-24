"""
narrative_generator.py – Philosophical & Analytical Storyteller.

This module converts raw probabilities, entropy scores, and model consensus
into human-readable "Investment Memos" or "War Room Briefings".
"""
from typing import Dict, Any

class NarrativeGenerator:
    """
    Generates narrative explanations for betting decisions.
    """

    @staticmethod
    def generate_story(match_data: Dict[str, Any], prediction: Dict[str, Any], entropy: float) -> str:
        """
        Constructs a narrative based on quantitative metrics.

        Args:
            match_data: Basic match info (Home, Away, League).
            prediction: Output from EnsembleModel (probs, consensus_score, details).
            entropy: Normalized entropy (0.0 - 1.0) from EntropyKelly.
        """
        home = match_data.get("home_team", "Home")
        away = match_data.get("away_team", "Away")

        prob_h = prediction.get("prob_home", 0.0)
        prob_a = prediction.get("prob_away", 0.0)
        consensus = prediction.get("consensus_score", 0.0)

        # Determine the favorite
        if prob_h > prob_a and prob_h > 0.34:
            fav = home
            prob = prob_h
        elif prob_a > prob_h and prob_a > 0.34:
            fav = away
            prob = prob_a
        else:
            fav = "Draw"
            prob = prediction.get("prob_draw", 0.0)

        # 1. Consensus Narrative
        if consensus > 0.8:
            consensus_text = "Models are in strong agreement."
        elif consensus > 0.5:
            consensus_text = "Models show moderate consensus."
        else:
            consensus_text = "High divergence among models (Chaos)."

        # 2. Entropy Narrative
        if entropy < 0.5:
            entropy_text = "Market uncertainty is low."
        elif entropy < 0.8:
            entropy_text = "Standard market uncertainty."
        else:
            entropy_text = "Extreme uncertainty (Coin Flip territory)."

        # 3. Model Specifics (Why did they vote this way?)
        details = prediction.get("details", {})
        reasons = []

        if "benter" in details:
            b_res = details["benter"]
            if b_res.get("prob_home", 0) > 0.5:
                reasons.append("Benter sees value in Home (Context/Form).")
            elif b_res.get("prob_away", 0) > 0.5:
                reasons.append("Benter favors Away.")

        if "dixon_coles" in details:
            dc_res = details["dixon_coles"]
            # Check if DC is significantly different from Benter?
            pass

        reason_text = " ".join(reasons)

        # 4. Final Story
        story = (
            f"🧠 **Investment Memo: {home} vs {away}**\n\n"
            f"**Outlook:** {fav} ({prob:.1%})\n"
            f"**Consensus:** {consensus_text} (Score: {consensus:.2f})\n"
            f"**Entropy:** {entropy_text} (H: {entropy:.2f})\n\n"
            f"**Rationale:**\n{reason_text}\n"
        )

        if entropy > 0.85:
            story += "\n⚠️ **Warning:** High Entropy suggests reducing stake size."

        return story
