import unittest
import asyncio
from unittest.mock import MagicMock
from src.pipeline.core import create_default_pipeline
from src.system.container import container
from src.core.regime_kelly import KellyDecision, RegimeState

class TestNewPipeline(unittest.TestCase):
    def setUp(self):
        # Mock container services
        self.mock_prob_engine = MagicMock()
        self.mock_prob_engine.predict.return_value = {"prob_home": 0.6}
        container.register("prob_engine", self.mock_prob_engine)

        self.mock_kelly = MagicMock()
        self.mock_kelly.calculate.return_value = KellyDecision(
            approved=True,
            stake_amount=100.0,
            stake_pct=0.01,
            edge=0.05
        )
        self.mock_kelly.load_state.return_value = True
        self.mock_kelly.save_state.return_value = None
        container.register("regime_kelly", self.mock_kelly)

    def test_pipeline_creation(self):
        """Test that default pipeline includes all new stages."""
        engine = create_default_pipeline()
        stage_names = [s.name for s in engine.stages]

        expected_stages = [
            "ingestion",
            "features",
            "inference",
            "ensemble",
            "risk",
            "execution",
            "reporting"
        ]

        for name in expected_stages:
            self.assertIn(name, stage_names, f"Stage {name} missing from pipeline")

    def test_risk_stage_integration(self):
        """Test RiskStage consumes Ensemble results correctly."""
        from src.pipeline.stages.risk import RiskStage
        stage = RiskStage()

        context = {
            "ensemble_results": [
                {"match_id": "test_match", "prob_home": 0.6, "news_summary": "Good news"}
            ],
            "matches": MagicMock() # Mock dataframe iterator needed?
        }

        # Mock matches dataframe to return odds
        import polars as pl
        matches_df = pl.DataFrame({
            "home_team": ["TeamA"],
            "away_team": ["TeamB"],
            "home_odds": [2.0]
        })
        context["matches"] = matches_df

        # Override match_id logic in RiskStage test:
        # RiskStage constructs match_id from home_team + away_team
        # So "test_match" won't match "TeamA_TeamB" unless we adjust inputs.

        context["ensemble_results"][0]["match_id"] = "TeamA_TeamB"

        # Run stage
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(stage.execute(context))

        self.assertTrue(len(result["final_bets"]) > 0)
        # 15.0 comes from: 100.0 (Mock) * 0.3 (VolModulator low win rate) * 0.5 (Markowitz Risk Aversion)
        self.assertAlmostEqual(result["final_bets"][0]["stake"], 15.0, delta=1.0)
        self.assertEqual(result["final_bets"][0]["news_summary"], "Good news")

if __name__ == '__main__':
    unittest.main()
