from typing import Dict, Any, List
from pathlib import Path
from loguru import logger
import polars as pl

from src.pipeline.core import PipelineStage
from src.system.container import container
from src.core.regime_kelly import RegimeKelly, RegimeState
from src.system.config import settings

class RiskStage(PipelineStage):
    """Calculates risk-adjusted stakes using Regime Kelly and EVT."""

    def __init__(self):
        super().__init__("risk")
        self.kelly = container.get("regime_kelly")

        # Load Bankroll State
        self.state_file = settings.DATA_DIR / "bankroll_state.json"
        if self.kelly:
            self.kelly.load_state(self.state_file)

        # Optional Imports
        try:
            from src.quant.copula_risk import CopulaRiskAnalyzer
            self.copula = CopulaRiskAnalyzer()
        except ImportError:
            self.copula = None

        try:
            from src.quant.evt_risk_manager import EVTRiskManager
            self.evt = EVTRiskManager()
        except ImportError:
            self.evt = None

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        ensemble_decisions = context.get("ensemble_results", [])
        matches = context.get("matches", pl.DataFrame())

        if not ensemble_decisions or matches.is_empty():
            return {"final_bets": []}

        # Create Odds Map
        # match_id -> odds (assuming match_id = home_away)
        odds_map = {}
        for row in matches.iter_rows(named=True):
            mid = f"{row.get('home_team')}_{row.get('away_team')}"
            odds_map[mid] = row.get("home_odds", 0.0)

        bets = []
        for decision in ensemble_decisions:
            match_id = decision.get("match_id")
            prob_home = decision.get("prob_home", 0.0)

            # Get Odds
            odds_home = odds_map.get(match_id, 0.0)

            if odds_home <= 1.0:
                continue

            # Regime Detection (Mock or from previous stage)
            # Ideal: VolatilityAnalyzer output from InferenceStage
            regime = RegimeState(volatility_regime="calm")

            kelly_decision = self.kelly.calculate(
                probability=prob_home,
                odds=odds_home,
                match_id=match_id,
                regime=regime
            )

            if kelly_decision.approved:
                bets.append({
                    "match_id": match_id,
                    "selection": "HOME",
                    "stake": kelly_decision.stake_amount,
                    "confidence": prob_home,
                    "odds": odds_home,
                    "reason": f"Kelly Approved (Edge: {kelly_decision.edge:.2%})",
                    "regime": regime.volatility_regime,
                    "news_summary": decision.get("news_summary", "")
                })

        # Copula Filtering
        if self.copula and len(bets) > 1:
            try:
                report = self.copula.analyze_coupon(bets)
                if report.dangerous_pairs:
                    logger.warning(f"Copula detected risk: {report.dangerous_pairs}")
            except Exception as e:
                logger.error(f"Copula error: {e}")

        # EVT Adjustment
        if self.evt and bets:
            try:
                bets = self.evt.adjust_kelly_stakes(bets)
            except Exception as e:
                logger.error(f"EVT error: {e}")

        # Save State
        if self.kelly:
            self.kelly.save_state(self.state_file)

        return {"final_bets": bets}
