
import sys
from pathlib import Path
import pytest
import asyncio

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.quant.risk.risk_control_tower import RiskControlTower, RiskDecision
from src.pipeline.stages.execution import ExecutionStage
from src.extensions.market_god import MarketGod, GodSignal

@pytest.mark.asyncio
async def test_risk_veto_logic():
    """Test that RiskControlTower correctly rejects critical vetoes."""
    tower = RiskControlTower()

    # 1. Chaos Veto Scenario
    # Mock Regime Detector return (simulate via monkeypatch or mock object logic)
    # Since we can't easily mock internal components without extensive setup,
    # we'll rely on the logic structure we injected.

    # We will verify by inspecting the code logic directly via `tower.evaluate_bet`
    # passed with specific triggers if possible, or by mocking the internal `regime_detector`.

    class MockRegime:
        regime = "CHAOTIC"
        confidence = 0.9 # Trigger veto

    class MockDetector:
        def detect_regime(self, mid, hist): return MockRegime()

    tower.regime_detector = MockDetector()

    candidate = {"match_id": "test_match", "prob_home": 0.5, "odds": 2.0}
    decision = tower.evaluate_bet(candidate, {})

    assert decision.approved is False
    assert "Chaos Veto" in decision.rejection_reason
    print("Risk Veto (Chaos) Test Passed.")

def test_market_god_logic():
    """Test MarketGod signal generation."""
    god = MarketGod()

    # 1. Fix Detection (Draw < 1.90)
    odds = {"home": 2.5, "draw": 1.85, "away": 2.5}
    sig = god.consult("match_fix", odds, [])

    assert sig.signal_type == "FIX_DETECTED"
    assert sig.conviction > 0.8
    assert sig.override_models is True
    print("Market God (Fix Detection) Test Passed.")

@pytest.mark.asyncio
async def test_execution_sanity_check():
    """Test ExecutionStage sanity checks."""
    stage = ExecutionStage()

    # 1. Valid Bet
    valid_bet = {"match_id": "m1", "stake": 100, "odds": 2.0, "ev": 0.05}
    assert stage._sanity_check(valid_bet) is True

    # 2. Negative Stake
    bad_stake = {"match_id": "m2", "stake": -10, "odds": 2.0}
    assert stage._sanity_check(bad_stake) is False

    # 3. Impossible Odds
    bad_odds = {"match_id": "m3", "stake": 10, "odds": 0.5}
    assert stage._sanity_check(bad_odds) is False

    # 4. Negative EV (Standard)
    neg_ev = {"match_id": "m4", "stake": 10, "odds": 2.0, "ev": -0.1}
    assert stage._sanity_check(neg_ev) is False

    print("Execution Sanity Check Test Passed.")

if __name__ == "__main__":
    # Manually run async tests if pytest not called directly
    loop = asyncio.new_event_loop()
    loop.run_until_complete(test_risk_veto_logic())
    loop.run_until_complete(test_execution_sanity_check())
    test_market_god_logic()
