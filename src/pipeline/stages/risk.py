from typing import Dict, Any, List, Optional
import polars as pl
from loguru import logger

from src.pipeline.core import PipelineStage
from src.system.container import container
from src.core.portfolio_optimizer import PortfolioOptimizer, PortfolioBet
from src.quant.risk.risk_control_tower import RiskControlTower
from src.quant.analysis.narrative_engine import NarrativeEngine
from src.pipeline.context import BettingContext

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
                "is_biscuit": decision.get("is_biscuit", False),
                "motivation_mismatch": decision.get("motivation_mismatch", 0.0),
                "home_motivation": decision.get("home_motivation", 5.0),
                "away_motivation": decision.get("away_motivation", 5.0),
                "teleology_narrative": decision.get("teleology_narrative", ""),
                "sample_size": decision.get("sample_size", 100),
                "model_count": decision.get("model_count", 1),
                "recent_results": decision.get("recent_results", []),
                "brier_score": decision.get("brier_score", 0.1),
                "epistemic_uncertainty": decision.get("epistemic_uncertainty", 0.5),
                "meta_quality_score": decision.get("meta_quality_score", 0.5),
                "board_multiplier": decision.get("board_multiplier", 1.0),
                "epistemic_multiplier": decision.get("epistemic_multiplier", 1.0),
                "hawkes_multiplier": decision.get("hawkes_multiplier", 1.0),
                "sm_multiplier": decision.get("sm_multiplier", 1.0),
                "gt_multiplier": decision.get("gt_multiplier", 1.0),
                "fractal_mult": decision.get("fractal_mult", 1.0),
                "game_theory_status": decision.get("game_theory_status", "NASH_EQUILIBRIUM")
            }

            # Mock open_bets if not present to enable CoVaR/Stress testing
            if "open_bets" not in context:
                context["open_bets"] = [{
                    "match_id": "locked_capital",
                    "odds": 2.0,
                    "prob_home": 0.5,
                    "ev": 0.0,
                    "stake_amount": self.tower.treasury.state.locked_capital
                }]

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
        # Determine global regime from the first approved bet (assuming regime is global)
        global_regime = "STABLE"
        if bet_metadata:
             first_meta = next(iter(bet_metadata.values()))
             if first_meta["risk_decision"].regime_metrics:
                 global_regime = first_meta["risk_decision"].regime_metrics.regime

        # Run Black-Litterman optimization globally on candidates
        from src.core.black_litterman_optimizer import BlackLittermanOptimizer
        bl_opt = BlackLittermanOptimizer()

        # Format candidates for BL
        bl_candidates = []
        for c in candidates:
            # We need standard dicts for BL Optimizer
            bl_candidates.append({
                "match_id": c.match_id,
                "selection": c.selection,
                "ev_home": c.ev,
                "best_ev": c.ev,
                "confidence": bet_metadata[c.match_id]["confidence"],
                "odds": c.odds,
                "league": c.league
            })

        bl_results = bl_opt.optimize(bl_candidates, {"regime": global_regime})

        # Apply BL multipliers/weights back to candidates before Markowitz
        for i, res in enumerate(bl_results):
            bl_weight = res.get("bl_weight", 0)
            if bl_weight <= 0:
                # Black Litterman vetoed it
                candidates[i].stake_pct = 0.0
                bet_metadata[candidates[i].match_id]["risk_decision"].approved = False
                existing_reason = bet_metadata[candidates[i].match_id]["risk_decision"].rejection_reason or ""
                bet_metadata[candidates[i].match_id]["risk_decision"].rejection_reason = existing_reason + " (Vetoed by Black-Litterman)"
            else:
                # BL suggests an optimal weight. We'll pass it to Markowitz as a hint
                # Or we can just use it directly to scale
                candidates[i].stake_pct = min(candidates[i].stake_pct, bl_weight)

        # Filter out rejected bets before sending to portfolio optimizer
        approved_candidates = [c for c in candidates if bet_metadata[c.match_id]["risk_decision"].approved]

        optimized_results = self.portfolio_opt.optimize(approved_candidates, regime=global_regime)

        final_bets = []
        # Check global God Signal from context if available
        god_signal = context.get("god_signal")
        is_black_swan = False
        if god_signal:
            if hasattr(god_signal, 'signal_type'):
                is_black_swan = god_signal.signal_type == "BLACK_SWAN"
            elif isinstance(god_signal, dict):
                is_black_swan = god_signal.get("signal_type") == "BLACK_SWAN"
            elif isinstance(god_signal, str):
                is_black_swan = god_signal == "BLACK_SWAN"

        if is_black_swan:
            logger.warning("Market God declared BLACK_SWAN. Canceling all final stakes to preserve capital.")

        for res in optimized_results:
            match_id = res["match_id"]
            meta = bet_metadata.get(match_id, {})
            risk_dec = meta.get("risk_decision")

            final_stake = res["stake_amount"] # From Portfolio Opt

            # Apply Black Swan override
            if is_black_swan:
                final_stake = 0.0

            # Apply God Multiplier from individual prediction if present
            # We can find the original prediction to see if there's a match-specific god_multiplier
            match_pred = next((d for d in ensemble_decisions if d.get("match_id") == match_id), None)
            if match_pred and match_pred.get("god_multiplier", 1.0) != 1.0:
                mult = match_pred.get("god_multiplier")
                final_stake *= mult
                logger.info(f"Applied God Multiplier ({mult}x) to {match_id} stake.")

            # Skip if stake was zeroed out
            if final_stake <= 0.0:
                continue
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
