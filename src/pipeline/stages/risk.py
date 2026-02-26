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
from src.quant.risk.volatility_modulator import VolatilityModulator
from src.core.cognitive_guardian import CognitiveGuardian
from src.quant.analysis.pre_mortem import PreMortemAnalyzer
from src.quant.finance.treasury import TreasuryEngine
from src.core.speed_cache import SpeedCache
from src.quant.risk.physics_modulator import PhysicsRiskModulator
from src.quant.finance.hedgehog import HedgeHog

# Enterprise Modules
try:
    from src.quant.analysis.philosophical_engine import PhilosophicalEngine, EpistemicReport
except ImportError:
    PhilosophicalEngine = None

try:
    from src.quant.risk.volatility_analyzer import VolatilityAnalyzer, VolatilityReport
except ImportError:
    VolatilityAnalyzer = None

try:
    from src.quant.analysis.narrative_engine import NarrativeEngine
except ImportError:
    NarrativeEngine = None

try:
    from src.pipeline.context import BettingContext
except ImportError:
    BettingContext = None

class RiskStage(PipelineStage):
    """
    Advanced Risk Stage (Level 42).
    Integrates Volatility (GARCH), Philosophy (Epistemic), Narrative (Voice),
    and Markowitz Portfolio Optimization.
    Now includes ALL 10 Advanced Physics Engines.
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

        # New Advanced Engines
        self.vol_modulator = VolatilityModulator()
        self.guardian = CognitiveGuardian()
        self.pre_mortem = PreMortemAnalyzer()
        self.treasury = TreasuryEngine()
        self.speed_cache = SpeedCache()
        self.physics_modulator = PhysicsRiskModulator()
        self.hedgehog = HedgeHog()

        # Optional Legacy Engines
        try:
            from src.quant.risk.copula_risk import CopulaRiskAnalyzer
            self.copula = CopulaRiskAnalyzer()
        except ImportError:
            self.copula = None

        try:
            from src.quant.risk.evt_risk_manager import EVTRiskManager
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

        # Global Systemic Risk Check via Ricci Flow
        ricci_report = context.get("ricci_report")

        # Physics Reports Map
        physics_reports = context.get("physics_reports", {})
        chaos_reports = physics_reports.get("chaos_reports", context.get("chaos_reports", {}))
        fractal_reports = physics_reports.get("fractal_reports", {})
        topology_reports = physics_reports.get("topology_reports", {})
        path_signatures = physics_reports.get("path_signatures", {})
        homology_reports = physics_reports.get("homology_reports", {})

        for decision in ensemble_decisions:
            match_id = decision.get("match_id")
            prob_home = decision.get("prob_home", 0.0)
            confidence = decision.get("confidence", 0.8)
            odds_home = odds_map.get(match_id, 0.0)

            if odds_home <= 1.0:
                continue

            # --- 0. Advanced Physics Modulation (Refactored) ---
            # Construct Physics Context
            physics_context = {}

            # Populate Context
            if match_id in chaos_reports:
                physics_context["chaos_regime"] = chaos_reports[match_id].regime

            if match_id in topology_reports:
                topo = topology_reports[match_id]
                physics_context["topology_anomaly"] = topo.is_anomalous
                physics_context["topology_cluster"] = topo.assigned_cluster

            if match_id in fractal_reports:
                frac = fractal_reports[match_id]
                physics_context["fractal_regime"] = frac.regime
                physics_context["fractal_dim"] = frac.fractal_dimension
                physics_context["fractal_mult"] = frac.kelly_multiplier # Legacy support

            if match_id in path_signatures:
                physics_context["roughness"] = path_signatures[match_id].get("sig_roughness", 0.0)

            if match_id in homology_reports:
                homo = homology_reports[match_id]
                physics_context["homology_org_diff"] = homo.get("org_advantage", 0.0)
                physics_context["home_org"] = homo.get("home_org", 0.0)
                physics_context["away_panicking"] = homo.get("away_panicking", False)

            # Apply Modulation
            physics_multiplier = self.physics_modulator.modulate(decision, physics_context, ricci_report)

            if physics_multiplier <= 0.0:
                logger.warning(f"Physics Kill Signal for {match_id}")
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

                # Direct Epistemic Scaling: If we don't "know" enough, we bet less.
                if philo_report.epistemic_approved:
                    epistemic_score = philo_report.epistemic_score
                else:
                    epistemic_score = 0.0 # If rejected, score is effectively 0 for multiplier logic

                if ctx:
                    ctx.philosophical_reports[match_id] = philo_report

            # --- C. Kelly Calculation (The Math) ---
            kelly_decision = self.kelly.calculate(
                probability=prob_home,
                odds=odds_home,
                match_id=match_id,
                regime=regime,
                epistemic_multiplier=epistemic_score # Passed directly to Kelly as multiplier
            )

            # --- C2. Advanced Modulators (The Titan Upgrade) ---
            if kelly_decision.approved:
                # 1. Meta-Labeling Check
                meta_score = decision.get("meta_quality_score", 0.5)
                if meta_score < 0.3:
                    kelly_decision.approved = False
                    kelly_decision.rejection_reason = f"Meta-Labeler Rejected (Score: {meta_score:.2f})"

                # 2. Volatility Scaling
                vol_mult = self.vol_modulator.get_kelly_fraction()
                kelly_decision.stake_pct *= vol_mult

                # 2.1 Physics Scaling
                kelly_decision.stake_pct *= physics_multiplier

                # 3. HedgeHog Logic (Real-time Hedging)
                # Check SpeedCache for real-time odds diffs (e.g. from API Hijacker stream)
                latest_odds = self.speed_cache.get(f"odds_{match_id}")
                if latest_odds:
                    # Construct a mock position to check for opportunity
                    # Assuming we are about to enter this position
                    mock_pos = {
                        "match_id": match_id,
                        "selection": "HOME",
                        "stake": kelly_decision.stake_amount, # Estimate
                        "odds": odds_home
                    }

                    # NOTE: Here we usually check existing positions, but we can also check if
                    # the current opportunity has drifted so much that we should wait or arbitrage.
                    # For simplicity, if odds drifted favourably (higher) before we bet, we might bet more?
                    # Or if they dropped (lower), we missed the boat?

                    if abs(latest_odds - odds_home) > 0.2:
                        logger.warning(f"HedgeHog: Real-time odds shift for {match_id} ({odds_home} -> {latest_odds})")
                        # If odds dropped significantly (value gone?), maybe reduce stake
                        if latest_odds < odds_home * 0.9:
                            kelly_decision.stake_pct *= 0.8
                            logger.info("HedgeHog: Odds dropped before execution. Reducing stake.")

                # 4. Cognitive Check
                bet_req = {"stake": kelly_decision.stake_pct, "team": decision.get("home_team")}
                if not self.guardian.check_bet(bet_req):
                    kelly_decision.approved = False
                    kelly_decision.rejection_reason = "Cognitive Guardian Blocked (Tilt/Chase)"

                # 5. Pre-Mortem Check (The Devil's Advocate)
                pm_report = self.pre_mortem.analyze({
                    "match_id": match_id,
                    "odds": odds_home,
                    "prob_model": prob_home,
                    "ev": kelly_decision.edge,
                    "selection": "HOME",
                    "home_odds": odds_home,
                    # We can fetch draw/away odds if available in full context or ignore for now
                    "draw_odds": matches.filter(pl.col("match_id") == match_id).select("draw_odds").item() if "draw_odds" in matches.columns else 0.0,
                    "away_odds": matches.filter(pl.col("match_id") == match_id).select("away_odds").item() if "away_odds" in matches.columns else 0.0,
                })

                if pm_report.kill_signal:
                    kelly_decision.approved = False
                    kelly_decision.rejection_reason = f"Pre-Mortem KILL: {', '.join(pm_report.reasons)}"
                elif pm_report.caution_signal:
                    kelly_decision.stake_pct *= 0.5
                    logger.warning(f"Pre-Mortem Caution {match_id}: Stake reduced 50%. ({pm_report.reasons})")

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
                    "kelly_decision": kelly_decision,
                    "physics": {
                        "chaos": physics_context.get('chaos_regime', "unknown"),
                        "fractal": physics_context.get('fractal_dim', 0.0),
                        "physics_mult": physics_multiplier,
                        "roughness": physics_context.get("roughness", 0.0),
                        "org_diff": physics_context.get("homology_org_diff", 0.0)
                    },
                    "pre_mortem_issues": pm_report.reasons if 'pm_report' in locals() and not pm_report.is_clean else []
                }
            else:
                logger.debug(f"Bet rejected for {match_id}: {kelly_decision.rejection_reason}")

        # --- E. Portfolio Optimization (The Finance) ---
        optimized_results = self.portfolio_opt.optimize(candidates)

        final_bets = []
        for res in optimized_results:
            match_id = res["match_id"]
            meta = bet_metadata.get(match_id, {})

            # --- Treasury Check ---
            # Ask Treasury for capital. It might reduce stake or deny based on daily drawdown.
            # Convert pct to amount (assuming base capital from treasury state)
            # Actually portfolio_opt output is amount or pct? Usually optimizer outputs weights.
            # Assuming 'stake_amount' is already calculated by PortfolioOptimizer based on some total capital.
            # We re-verify with Treasury.

            requested_stake = res["stake_amount"]
            approved_stake = self.treasury.request_capital(requested_stake, strategy_type="aggressive") # Defaulting to aggressive for now

            if approved_stake < requested_stake:
                logger.warning(f"Treasury reduced stake for {match_id}: {requested_stake:.2f} -> {approved_stake:.2f}")
                res["stake_amount"] = approved_stake

            if approved_stake <= 0:
                continue

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
                # Append Physics Info to Narrative
                phy = meta.get("physics", {})
                narrative += f"\n\n⚛️ **Deep Physics**\n- Chaos: {phy.get('chaos')}\n- Fractal Dim: {phy.get('fractal'):.3f}"
                if phy.get("roughness", 0) > 0:
                    narrative += f"\n- Path Roughness: {phy.get('roughness'):.3f}"
                if phy.get("physics_mult", 1.0) < 1.0:
                    narrative += f"\n- Physics Modulation: {phy.get('physics_mult'):.2f}x"

                pm_issues = meta.get("pre_mortem_issues", [])
                if pm_issues:
                    narrative += f"\n\n⚠️ **Pre-Mortem Warning**\n" + "\n".join([f"- {i}" for i in pm_issues])

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
