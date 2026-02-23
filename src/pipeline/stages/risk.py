from typing import Dict, Any, List, Optional
import numpy as np
import polars as pl
from loguru import logger
import hashlib

from src.pipeline.core import PipelineStage
from src.system.container import container
from src.core.regime_kelly import RegimeKelly, RegimeState
from src.core.portfolio_optimizer import PortfolioOptimizer, PortfolioBet
from src.system.config import settings

# Enterprise Modules
try:
    from src.quant.philosophical_engine import PhilosophicalEngine, EpistemicReport
except ImportError:
    PhilosophicalEngine = None

try:
    from src.quant.risk import VolatilityAnalyzer, VolatilityReport
except ImportError:
    VolatilityAnalyzer = None

try:
    from src.quant.narrative_engine import NarrativeEngine
except ImportError:
    NarrativeEngine = None

try:
    from src.pipeline.context import BettingContext
except ImportError:
    BettingContext = None

class RiskStage(PipelineStage):
    """
    Advanced Risk Stage (Level 41).
    Integrates Volatility (GARCH), Philosophy (Epistemic), Narrative (Voice),
    and Markowitz Portfolio Optimization.
    """

    def __init__(self):
        super().__init__("risk")
        self.kelly = container.get("regime_kelly")
        self.portfolio_opt = container.get("portfolio_opt") or PortfolioOptimizer()

        # Load Bankroll State
        self.state_file = settings.DATA_DIR / "bankroll_state.json"
        if self.kelly:
            self.kelly.load_state(self.state_file)

        # Initialize Engines
        self.philosopher = PhilosophicalEngine() if PhilosophicalEngine else None
        self.vol_analyzer = VolatilityAnalyzer() if VolatilityAnalyzer else None
        self.narrator = NarrativeEngine() if NarrativeEngine else None

        # Optional Legacy Engines
        try:
            from src.quant.risk import CopulaRiskAnalyzer
            self.copula = CopulaRiskAnalyzer()
        except ImportError:
            self.copula = None

        try:
            from src.quant.risk import EVTRiskManager
            self.evt = EVTRiskManager()
        except ImportError:
            self.evt = None

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute risk analysis pipeline."""

        # 1. Retrieve Context
        ctx: Optional[BettingContext] = context.get("ctx")
        if not ctx and BettingContext:
             # Fallback: create context from dict
             ctx = BettingContext.from_dict(context)

        ensemble_decisions = context.get("ensemble_results", [])
        matches = context.get("matches", pl.DataFrame())

        if not ensemble_decisions or matches.is_empty():
            logger.info("No decisions to process in RiskStage.")
            return {"final_bets": []}

        # Match ID -> Odds Map
        odds_map = {}
        for row in matches.iter_rows(named=True):
            mid = f"{row.get('home_team')}_{row.get('away_team')}"
            odds_map[mid] = row.get("home_odds", 0.0)

        # Candidate Bets for Portfolio Optimization
        candidates: List[PortfolioBet] = []
        # Temporary storage to link back candidates to full bet info
        bet_metadata = {}

        for decision in ensemble_decisions:
            match_id = decision.get("match_id")
            prob_home = decision.get("prob_home", 0.0)
            confidence = decision.get("confidence", 0.8)
            odds_home = odds_map.get(match_id, 0.0)

            if odds_home <= 1.0:
                continue

            # --- A. Volatility Analysis (The Context) ---
            vol_report = None
            regime = RegimeState(volatility_regime="calm") # Default

            if self.vol_analyzer:
                sim_returns = self._simulate_returns(match_id)
                vol_report = self.vol_analyzer.analyze(
                    sim_returns, match_id=match_id, market="odds"
                )
                regime.volatility_regime = vol_report.regime
                if ctx:
                    ctx.volatility_reports[match_id] = vol_report

            # --- B. Philosophical Analysis (The Wisdom) ---
            philo_report = None
            epistemic_score = 1.0

            if self.philosopher:
                spread = self._calculate_spread(matches, match_id)
                philo_report = self.philosopher.evaluate(
                    probability=prob_home,
                    confidence=confidence,
                    sample_size=200,
                    match_id=match_id,
                    market_odds_spread=spread
                )
                epistemic_score = philo_report.epistemic_score if philo_report.epistemic_approved else 0.0
                if ctx:
                    ctx.philosophical_reports[match_id] = philo_report

            # --- C. Kelly Calculation (The Math) ---
            kelly_decision = self.kelly.calculate(
                probability=prob_home,
                odds=odds_home,
                match_id=match_id,
                regime=regime,
                epistemic_multiplier=epistemic_score
            )

            # --- D. Portfolio Candidates ---
            if kelly_decision.approved:
                # Create PortfolioBet candidate
                pb = PortfolioBet(
                    match_id=match_id,
                    selection="HOME",
                    odds=odds_home,
                    prob_model=prob_home,
                    ev=kelly_decision.edge,
                    stake_pct=kelly_decision.stake_pct,
                    league="Unknown", # Could be fetched from matches DF
                    correlation_group="Standard"
                )
                candidates.append(pb)

                # Store metadata for narrative generation later
                bet_metadata[match_id] = {
                    "confidence": confidence,
                    "philo_report": philo_report,
                    "vol_report": vol_report,
                    "news_summary": decision.get("news_summary", ""),
                    "kelly_decision": kelly_decision
                }
            else:
                logger.debug(f"Bet rejected for {match_id}: {kelly_decision.rejection_reason}")

        # --- E. Portfolio Optimization (The Finance) ---
        optimized_results = self.portfolio_opt.optimize(candidates)

        final_bets = []
        for res in optimized_results:
            match_id = res["match_id"]
            meta = bet_metadata.get(match_id, {})

            # Generate Narrative with finalized stake
            narrative = ""
            if self.narrator:
                narrative = self.narrator.generate_memo(
                    match_id=match_id,
                    selection=res["selection"],
                    odds=res["odds"],
                    stake=res["stake_amount"],
                    confidence=meta.get("confidence", 0.0),
                    edge=res["ev"],
                    philo_report=meta.get("philo_report"),
                    vol_report=meta.get("vol_report"),
                    news_summary=meta.get("news_summary", "")
                )
                if ctx:
                    ctx.narratives[match_id] = narrative

            bet_record = {
                "match_id": match_id,
                "selection": res["selection"],
                "stake": res["stake_amount"],
                "confidence": meta.get("confidence", 0.0),
                "odds": res["odds"],
                "reason": f"Markowitz Optimized (Kelly: {res['raw_stake_pct']:.2%} -> Adj: {res['adjusted_stake_pct']:.2%})",
                "regime": meta.get("vol_report").regime if meta.get("vol_report") else "unknown",
                "news_summary": meta.get("news_summary", ""),
                "narrative": narrative,
                "timestamp": meta.get("kelly_decision").timestamp if meta.get("kelly_decision") else "",
                "trading_mode": res.get("trading_mode", "LIVE"),
                "is_paper": res.get("is_paper", False)
            }
            final_bets.append(bet_record)

        # --- F. Legacy Checks (Copula/EVT) ---
        # These operate on final list. Note: PortfolioOptimizer already handles correlation,
        # so Copula might be redundant but kept for safety.
        if self.copula and len(final_bets) > 1:
            try:
                report = self.copula.analyze_coupon(final_bets)
                if report.dangerous_pairs:
                    logger.warning(f"Copula detected risk: {report.dangerous_pairs}")
            except Exception as e:
                logger.error(f"Copula error: {e}")

        # Save State
        if self.kelly:
            self.kelly.save_state(self.state_file)

        # Update Context
        if ctx:
            ctx.final_bets = final_bets

        return {"final_bets": final_bets, "ctx": ctx}

    def _simulate_returns(self, match_id: str) -> np.ndarray:
        """
        Simulates returns for Volatility Analysis when historical DB is unavailable.
        Uses hash of match_id to be deterministic.
        """
        seed = int(hashlib.md5(match_id.encode()).hexdigest(), 16) % 2**32
        np.random.seed(seed)
        n = 100
        returns = np.random.normal(0, 0.01, n)
        if seed % 3 == 0:
            returns[-10:] *= 3 # Recent volatility spike
        elif seed % 5 == 0:
            returns[-20:-10] *= 2 # Past volatility spike
        return returns

    def _calculate_spread(self, matches: pl.DataFrame, match_id: str) -> float:
        """Calculates market spread from odds."""
        try:
            parts = match_id.split("_")
            if len(parts) >= 2:
                home = parts[0]
                away = parts[1]
                row = matches.filter(
                    (pl.col("home_team") == home) & (pl.col("away_team") == away)
                ).head(1)
                if not row.is_empty():
                    h = row["home_odds"][0]
                    d = row["draw_odds"][0]
                    a = row["away_odds"][0]
                    if h > 0 and d > 0 and a > 0:
                        implied_prob = (1/h) + (1/d) + (1/a)
                        spread = implied_prob - 1.0
                        return max(spread, 0.0)
        except Exception:
            pass
        return 0.05 # Default 5% spread
