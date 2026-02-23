from typing import Dict, Any, List
from loguru import logger
import polars as pl
from src.pipeline.core import PipelineStage
from src.system.container import container
from src.core.regime_kelly import RegimeKelly, RegimeState

class RiskStage(PipelineStage):
    """Calculates risk-adjusted stakes using Regime Kelly and EVT."""

    def __init__(self):
        super().__init__("risk")
        self.kelly = container.get("regime_kelly")

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
        preds = context.get("ensemble", [])
        matches = context.get("matches", pl.DataFrame())

        if not preds or matches.is_empty():
            return {"final_bets": []}

        bets = []
        for i, pred in enumerate(preds):
            # Assuming pred is a dict with prob_home, prob_draw, prob_away
            row = matches.row(i, named=True)
            home, away = row["home_team"], row["away_team"]

            # Simple logic: check highest edge
            # For demonstration, only checking home win
            prob_home = pred.get("prob_home", 0.0)
            odds_home = row.get("home_odds", 0.0)

            # Regime Detection (Mock or from previous stage)
            regime = RegimeState(volatility_regime="calm") # Should come from VolatilityAnalyzer

            decision = self.kelly.calculate(
                probability=prob_home,
                odds=odds_home,
                match_id=f"{home}_{away}",
                regime=regime
            )

            if decision.approved:
                bets.append({
                    "match_id": decision.match_id,
                    "selection": "HOME",
                    "stake": decision.stake_amount,
                    "confidence": prob_home,
                    "odds": odds_home,
                    "reason": "Regime Kelly Approved"
                })

        # Copula Filtering
        if self.copula and len(bets) > 1:
            report = self.copula.analyze_coupon(bets)
            if report.dangerous_pairs:
                logger.warning(f"Copula detected risk: {report.dangerous_pairs}")
                # Filter logic would go here

        # EVT Adjustment
        if self.evt and bets:
            bets = self.evt.adjust_kelly_stakes(bets)

        return {"final_bets": bets}
