from typing import Dict, Any, List, Optional
import numpy as np
import polars as pl
from loguru import logger

from src.pipeline.core import PipelineStage
from src.system.container import container
from src.core.portfolio_optimizer import PortfolioOptimizer, PortfolioBet
from src.quant.risk.risk_control_tower import RiskControlTower
from src.quant.analysis.narrative_engine import NarrativeEngine
from src.pipeline.context import BettingContext

try:
    from src.quant.analysis.philosophical_engine import PhilosophicalEngine
except ImportError:
    PhilosophicalEngine = None

class RiskStage(PipelineStage):
    """
    Refactored Risk Stage (Level 43).
    Delegates complex risk logic to the central `RiskControlTower`.
    """

    def __init__(self):
        super().__init__("risk")
        # Central Authority
        self.tower = RiskControlTower()

        self.portfolio_opt = container.get("portfolio_opt") or PortfolioOptimizer()
        self.narrator = NarrativeEngine()

        # Legacy Support (Copula/EVT) if needed, but Tower should handle most.
        try:
            from src.quant.risk.copula_risk import CopulaRiskAnalyzer
            self.copula = CopulaRiskAnalyzer()
        except ImportError:
            self.copula = None

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute risk analysis pipeline."""

        # 1. Retrieve Context
        ctx: Optional[BettingContext] = context.get("ctx")
        if not ctx and BettingContext:
             ctx = BettingContext.from_dict(context)

        ensemble_decisions = context.get("ensemble_results", [])
        matches = context.get("matches", pl.DataFrame())

        if not ensemble_decisions or matches.is_empty():
            logger.info("No decisions to process in RiskStage.")
            return {"final_bets": []}

        # Match ID -> Odds Map
        odds_map = {}
        for row in matches.iter_rows(named=True):
            mid = row["match_id"]
            odds_map[mid] = row.get("home_odds", 0.0)

        # Candidate Bets for Portfolio Optimization
        candidates: List[PortfolioBet] = []
        bet_metadata = {}

        for decision in ensemble_decisions:
            match_id = decision.get("match_id")
            prob_home = decision.get("prob_home", 0.0)
            confidence = decision.get("confidence", 0.8)
            odds_home = odds_map.get(match_id, 0.0)

            if odds_home <= 1.0:
                continue

            # Construct Bet Candidate Dict for Tower
            bet_candidate = {
                "match_id": match_id,
                "home_team": decision.get("home_team"),
                "prob_home": prob_home,
                "odds": odds_home,
                "confidence": confidence,
                "ev": (prob_home * odds_home) - 1.0,
                # Pass other needed data
            }

            # --- DELEGATE TO TOWER ---
            # The Tower handles Cognitive, PreMortem, Kelly, Physics, Treasury.
            risk_decision = self.tower.evaluate_bet(bet_candidate, context)

            if risk_decision.approved:
                # Add to Portfolio Candidates
                pb = PortfolioBet(
                    match_id=match_id,
                    selection="HOME",
                    odds=odds_home,
                    prob_model=prob_home,
                    ev=bet_candidate["ev"],
                    stake_pct=risk_decision.stake_pct,
                    league="Unknown", # Extract from context if available
                    correlation_group="Standard"
                )
                candidates.append(pb)

                # Store metadata for narrative generation
                bet_metadata[match_id] = {
                    "risk_decision": risk_decision,
                    "confidence": confidence,
                    "news_summary": decision.get("news_summary", ""),
                    "teleology": {
                        "score": decision.get("teleology_score", 0.5),
                        "narrative": decision.get("teleology_narrative", ""),
                        "is_biscuit": decision.get("is_biscuit", False)
                    }
                }
            else:
                logger.info(f"Bet Rejected ({match_id}): {risk_decision.rejection_reason}")

        # --- Portfolio Optimization ---
        # Markowitz over the approved candidates
        optimized_results = self.portfolio_opt.optimize(candidates)

        final_bets = []
        for res in optimized_results:
            match_id = res["match_id"]
            meta = bet_metadata.get(match_id, {})
            risk_dec = meta.get("risk_decision")

            # Final check: Treasury has already approved the max stake in Tower.
            # Portfolio Optimization might reduce it further for diversification.
            # We trust the lower of the two.

            final_stake = res["stake_amount"] # From Portfolio Opt
            # Ensure it doesn't exceed what Treasury approved (should be enforced by stake_pct in candidate)

            # Generate Narrative
            narrative = self.narrator.generate_memo(
                match_id=match_id,
                selection=res["selection"],
                odds=res["odds"],
                stake=final_stake,
                confidence=meta.get("confidence", 0.0),
                edge=res["ev"],
                news_summary=meta.get("news_summary", "")
            )

            # Append Tower Rationale
            if risk_dec:
                narrative += f"\n\n🛡️ **Risk Tower**: {risk_dec.rationale}"
                if risk_dec.regime_metrics:
                    narrative += f"\nMarket Regime: {risk_dec.regime_metrics.regime}"

            bet_record = {
                "match_id": match_id,
                "selection": res["selection"],
                "stake": final_stake,
                "confidence": meta.get("confidence", 0.0),
                "odds": res["odds"],
                "reason": f"Tower + Markowitz. {risk_dec.rejection_reason}",
                "narrative": narrative,
                "timestamp": "", # Add timestamp
                "trading_mode": res.get("trading_mode", "LIVE"),
                "is_paper": res.get("is_paper", False)
            }
            final_bets.append(bet_record)

            if ctx:
                ctx.narratives[match_id] = narrative

        # Update Context
        if ctx:
            ctx.final_bets = final_bets

        return {"final_bets": final_bets, "ctx": ctx}
