"""
teleology.py – Teleological Reasoning Engine (Purpose & Motivation).

Unlike statistical models that ask "What will happen?", Teleology asks "Why are they playing?".
It detects game-theoretic scenarios like "Biscuit Games" (mutually beneficial draws)
and motivation mismatches (Desperation vs. Mercenary).
"""
from typing import Dict, Any, Optional
from loguru import logger
import numpy as np

class TeleologicalEngine:
    """
    Analyzes the 'Purpose' of the match.
    """

    def __init__(self):
        logger.debug("TeleologicalEngine initialized.")

    def analyze(self, match_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entry point for teleological analysis.

        Args:
            match_context: Dict containing match info, points, standings, stage of season.
                           Expected keys: 'home_points', 'away_points', 'home_rank', 'away_rank',
                           'total_rounds', 'current_round', 'league_type' (league/cup).

        Returns:
            Dict with scores and narratives.
        """
        # Default safety values
        score = 0.5  # 0.5 = Neutral
        narrative = []

        # 1. Detect Biscuit Game (Mutually Beneficial Draw)
        biscuit_res = self.detect_biscuit_game(match_context)
        if biscuit_res["is_biscuit"]:
            score = 0.9 # High probability of specific outcome (Draw)
            narrative.append(f"🍪 **Biscuit Game Alert**: {biscuit_res['reason']}")

        # 2. Detect Motivation Mismatch
        motiv_res = self.detect_motivation_mismatch(match_context)
        if motiv_res["mismatch_score"] > 0.7:
            # Shift score towards the motivated team
            # If home is motivated (score > 0), increased prob.
            narrative.append(f"🔥 **Motivation Gap**: {motiv_res['reason']}")

        return {
            "teleology_score": score,
            "is_biscuit": biscuit_res["is_biscuit"],
            "motivation_mismatch": motiv_res["mismatch_score"],
            "narrative": " ".join(narrative) if narrative else "Standard competitive match."
        }

    def detect_biscuit_game(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detects if a draw suits both teams.
        Common in late-season league games or group stages.
        """
        current_round = ctx.get("current_round", 1)
        total_rounds = ctx.get("total_rounds", 38)

        # Only relevant in late season
        if current_round < total_rounds * 0.8:
            return {"is_biscuit": False, "reason": ""}

        # Logic: If both teams need 1 point to secure objective (Title, Europe, Safety)
        # This requires deep knowledge of the table.
        # Simplified Heuristic:
        # If both teams are neighbors in the table and near a cutoff zone (relegation/europe)
        # and odds for Draw are unusually low (< 2.5).

        h_rank = ctx.get("home_rank", 10)
        a_rank = ctx.get("away_rank", 11)
        draw_odds = ctx.get("draw_odds", 3.0)

        is_neighbor = abs(h_rank - a_rank) <= 2

        # Market signal for biscuit: Draw odds drop significantly
        market_signals_biscuit = draw_odds < 2.10

        if is_neighbor and market_signals_biscuit:
            return {
                "is_biscuit": True,
                "reason": "League neighbors with suspicious Draw odds."
            }

        return {"is_biscuit": False, "reason": ""}

    def detect_motivation_mismatch(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detects if one team cares significantly more than the other.
        """
        current_round = ctx.get("current_round", 1)
        total_rounds = ctx.get("total_rounds", 38)

        if current_round < total_rounds * 0.7:
            return {"mismatch_score": 0.0, "reason": ""}

        h_rank = ctx.get("home_rank", 10)
        a_rank = ctx.get("away_rank", 10)
        n_teams = ctx.get("n_teams", 20)

        # Zones
        relegation_zone = n_teams - 3
        europe_zone = 5
        title_zone = 2

        # Determine motivation level (0-10)
        def get_motivation(rank):
            if rank <= title_zone: return 10.0 # Title chase
            if rank >= relegation_zone: return 10.0 # Survival
            if rank <= europe_zone: return 8.0 # Europe
            return 2.0 # Mid-table (Flip-flops / Mercenary Mode)

        h_mot = get_motivation(h_rank)
        a_mot = get_motivation(a_rank)

        diff = abs(h_mot - a_mot)
        normalized_diff = diff / 10.0

        reason = ""
        if normalized_diff > 0.5:
            favored = "Home" if h_mot > a_mot else "Away"
            reason = f"{favored} is desperate/chasing, opponent is likely indifferent."

        return {
            "mismatch_score": normalized_diff,
            "reason": reason,
            "home_motivation": h_mot,
            "away_motivation": a_mot
        }
