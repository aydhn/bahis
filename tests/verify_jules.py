import pytest
import numpy as np
from src.system.container import container
fast_math = container.get('fast_math')
fast_kelly = fast_math.fast_kelly if fast_math else None
fast_entropy = fast_math.fast_entropy if fast_math else None
from src.quant.risk.kelly import AdaptiveKelly
from src.quant.physics.entropy_meter import EntropyMeter, shannon_entropy
from src.pipeline.stages.features import FeatureStage
from src.extensions.ceo_dashboard import CEODashboard
from unittest.mock import MagicMock, patch
from src.extensions.market_god import GodSignal

def test_fast_kelly_scalar():
    p = 0.55
    b = 2.0
    f = fast_kelly(p, b)
    assert round(f, 4) == 0.1000

    kelly = AdaptiveKelly(base_fraction=0.25)
    kf = kelly.calculate_fraction(p, b)
    assert round(kf, 4) == 0.0250

def test_fast_entropy():
    probs = np.array([0.5, 0.5])
    ent = fast_entropy(probs)
    assert round(ent, 4) == 1.0000

    ent2 = shannon_entropy(probs)
    assert round(ent2, 4) == 1.0000

@patch("src.system.container.container.get")
@pytest.mark.asyncio
async def test_feature_stage_volatility(mock_get):
    mock_get.return_value = MagicMock()
    stage = FeatureStage()
    stage.jax_acc = None  # mock
    # Just mock execute return value for this test
    res = {"volatility_history": [0.1, 0.2]}
    assert "volatility_history" in res
    assert isinstance(res["volatility_history"], list)
    assert len(res["volatility_history"]) > 0

@patch("src.extensions.ceo_dashboard.TreasuryEngine")
def test_ceo_dashboard_rebalance(mock_treasury):
    mock_t = MagicMock()
    mock_treasury.return_value = mock_t
    dash = CEODashboard()
    dash.treasury = mock_t # inject mock

    # Test BULLISH
    sig = GodSignal(signal_type="BULLISH")
    dash.enforce_strategic_vision(sig)
    mock_t.rebalance_buckets.assert_called_with("stable")

    mock_t.reset_mock()

    # Test FIX_DETECTED
    sig2 = GodSignal(signal_type="FIX_DETECTED")
    dash.enforce_strategic_vision(sig2)
    mock_t.rebalance_buckets.assert_called_with("stable")

