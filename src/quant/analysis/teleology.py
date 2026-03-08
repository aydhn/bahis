"""
teleology.py – Teleological Reasoning Engine (Purpose & Motivation).

Unlike statistical models that ask "What will happen?", Teleology asks "Why are they playing?".
It detects game-theoretic scenarios like "Biscuit Games" (mutually beneficial draws)
and motivation mismatches (Desperation vs. Mercenary).
"""
from typing import Dict, Any
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

        # 3. Analyze Narrative Momentum (The Visionary)
        narrative_mom = self.analyze_narrative_momentum(match_context)
        if narrative_mom["hype_score"] != 0:
            direction = "Home" if narrative_mom["hype_score"] > 0 else "Away"
            intensity = abs(narrative_mom["hype_score"])
            # Adjust score slightly (Narrative isn't everything)
            score += narrative_mom["hype_score"] * 0.1
            if intensity > 0.5:
                narrative.append(f"📰 **Narrative Momentum**: {direction} Hype (Score: {intensity:.2f})")

        return {
            "teleology_score": np.clip(score, 0.0, 1.0),
            "is_biscuit": biscuit_res["is_biscuit"],
            "motivation_mismatch": motiv_res["mismatch_score"],
            "home_motivation": motiv_res.get("home_motivation", 5.0),
            "away_motivation": motiv_res.get("away_motivation", 5.0),
            "narrative": " ".join(narrative) if narrative else "Standard competitive match."
        }

    def analyze_narrative_momentum(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulates "social sentiment" or "news hype" affecting team motivation.

        This mimics the "Visionary" aspect: detecting if a team is "on a mission"
        or "imploding" due to external narratives (Manager Sacked, Club Anniversary, etc.).

        Since we don't have a real news feed, we infer this from:
        - Recent Form vs Expectations (Underdog performing well?)
        - Derby status (Context flag)
        - Random Walk (Simulating unseen social media buzz)
        """
        match_id = ctx.get("match_id", "")
        # Deterministic simulation based on match_id hash
        import hashlib
        seed = int(hashlib.md5(match_id.encode()).hexdigest(), 16) % 100

        # 0 = Neutral, 1 = Max Hype Home, -1 = Max Hype Away
        hype_score = 0.0

        # 1. Random Narrative Event (10% chance)
        if seed < 10:
            # Random "Manager Sacked" bounce or "Club Crisis"
            hype_score = (seed - 5) / 5.0 # -1.0 to 0.8

        # 2. Underdog Narrative
        h_odds = ctx.get("home_odds", 2.0)
        a_odds = ctx.get("away_odds", 2.0)

        # If huge underdog (odds > 5.0) and hype is positive -> "Giant Killer" narrative
        if h_odds > 5.0 and hype_score > 0.2:
            hype_score += 0.3 # Boost narrative

        return {
            "hype_score": hype_score,
            "description": "Simulated Narrative Momentum"
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
