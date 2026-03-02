import sys
from unittest.mock import MagicMock

# Mock out heavy dependencies due to restricted environment per memory instructions
sys.modules['numpy'] = MagicMock()
sys.modules['scipy'] = MagicMock()
sys.modules['scipy.stats'] = MagicMock()
sys.modules['scipy.optimize'] = MagicMock()
sys.modules['scipy.special'] = MagicMock()
sys.modules['loguru'] = MagicMock()
sys.modules['polars'] = MagicMock()
sys.modules['numba'] = MagicMock()
sys.modules['torch'] = MagicMock()
sys.modules['torch.nn'] = MagicMock()
sys.modules['torch.optim'] = MagicMock()
sys.modules['torch.nn.functional'] = MagicMock()
sys.modules['cv2'] = MagicMock()

# Mock out pydantic config to avoid env errors
class MockSettings:
    DATA_DIR = MagicMock()

sys.modules['pydantic_settings'] = MagicMock()
sys.modules['src.system.config'] = MagicMock()
sys.modules['src.system.config'].settings = MockSettings()

import pytest
from src.quant.finance.treasury import TreasuryEngine, TreasuryState

def test_stress_test_portfolio(tmp_path):
    # Setup Treasury Engine with temporary path so it doesn't write to actual disk config
    engine = TreasuryEngine(state_path=tmp_path / "treasury_state.json")

    # ----------------------------------------------------
    # Case 1: SOLVENT
    # Total Cap: 10000.0, Locked: 1000.0, Shock: 0.20
    # Projected Loss = 200.0
    # Projected Cap = 9800.0
    # Remaining Locked = 800.0
    # Liquidity Ratio = (9800.0 - 800.0) / 9800.0 = 9000.0 / 9800.0 ≈ 0.918 (> 0.05)
    # ----------------------------------------------------
    engine.state.total_capital = 10000.0
    engine.state.locked_capital = 1000.0

    result = engine.stress_test_portfolio(shock_factor=0.20)

    assert result["status"] == "SOLVENT"
    assert result["shock_factor"] == 0.20
    assert result["projected_loss"] == 200.0
    assert result["projected_capital"] == 9800.0
    assert result["liquidity_ratio"] == round((9800.0 - 800.0) / 9800.0, 3)

    # ----------------------------------------------------
    # Case 2: ILLIQUID
    # Total Cap: 10000.0, Locked: 9800.0, Shock: 0.50
    # Projected Loss = 4900.0
    # Projected Cap = 5100.0
    # Remaining Locked = 4900.0
    # Liquidity Ratio = (5100.0 - 4900.0) / 5100.0 = 200.0 / 5100.0 ≈ 0.039 (< 0.05)
    # ----------------------------------------------------
    engine.state.total_capital = 10000.0
    engine.state.locked_capital = 9800.0

    result = engine.stress_test_portfolio(shock_factor=0.50)

    assert result["status"] == "ILLIQUID"
    assert result["shock_factor"] == 0.50
    assert result["projected_loss"] == 4900.0
    assert result["projected_capital"] == 5100.0
    assert result["liquidity_ratio"] == round((5100.0 - 4900.0) / 5100.0, 3)

    # ----------------------------------------------------
    # Case 3: INSOLVENT (Projected Capital <= 0)
    # Total Cap: 10000.0, Locked: 10000.0, Shock: 1.0 (100% loss)
    # Projected Loss = 10000.0
    # Projected Cap = 0.0
    # Liquidity Ratio = 0.0
    # ----------------------------------------------------
    engine.state.total_capital = 10000.0
    engine.state.locked_capital = 10000.0

    result = engine.stress_test_portfolio(shock_factor=1.0)

    assert result["status"] == "INSOLVENT"
    assert result["shock_factor"] == 1.0
    assert result["projected_loss"] == 10000.0
    assert result["projected_capital"] == 0.0
    assert result["liquidity_ratio"] == 0.0

    # ----------------------------------------------------
    # Case 4: INSOLVENT (Projected Capital < 0)
    # ----------------------------------------------------
    engine.state.total_capital = 10000.0
    engine.state.locked_capital = 15000.0  # Theoretically possible if overleveraged

    result = engine.stress_test_portfolio(shock_factor=0.80)
    # Projected Loss = 15000 * 0.8 = 12000
    # Projected Cap = 10000 - 12000 = -2000

    assert result["status"] == "INSOLVENT"
    assert result["shock_factor"] == 0.80
    assert result["projected_loss"] == 12000.0
    assert result["projected_capital"] == -2000.0
    assert result["liquidity_ratio"] == 0.0
