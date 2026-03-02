import sys
import math
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
sys.modules['pydantic_settings'] = MagicMock()

import pytest
from src.quant.finance.liquidity_engine import LiquidityEngine

@pytest.fixture
def engine():
    return LiquidityEngine()

def test_simulate_execution_zero_or_negative_stake(engine):
    """Test that zero or negative stakes return original odds and 0 slippage."""
    odds = 2.0
    league = "Default"

    # Zero stake
    exec_price, slippage = engine.simulate_execution(0.0, odds, league)
    assert exec_price == odds
    assert slippage == 0.0

    # Negative stake
    exec_price, slippage = engine.simulate_execution(-100.0, odds, league)
    assert exec_price == odds
    assert slippage == 0.0

def test_simulate_execution_small_stake_no_slippage(engine):
    """Test stakes that consume less than 1 tick return original odds."""
    odds = 2.0
    league = "Default"
    base_volume = engine.LEAGUE_LIQUIDITY[league] # 5000.0
    depth_per_tick = base_volume * 0.01 # 50.0

    # Stake exactly 1 tick minus a tiny amount
    stake = depth_per_tick - 0.001
    exec_price, slippage = engine.simulate_execution(stake, odds, league)
    assert exec_price == odds
    assert slippage == 0.0

    # Stake much smaller than 1 tick
    stake = 10.0
    exec_price, slippage = engine.simulate_execution(stake, odds, league)
    assert exec_price == odds
    assert slippage == 0.0

def test_simulate_execution_normal_stake(engine):
    """Test stakes that consume >= 1 tick and calculate exact avg price."""
    odds = 2.0
    league = "Default"
    base_volume = engine.LEAGUE_LIQUIDITY[league] # 5000.0
    depth_per_tick = base_volume * 0.01 # 50.0
    tick_size = 0.01

    # Stake consumes exactly 4 ticks
    stake = depth_per_tick * 4.0 # 200.0
    ticks_consumed = 4.0
    expected_avg_price = odds - (ticks_consumed / 2.0) * tick_size # 2.0 - 2.0 * 0.01 = 1.98
    expected_slippage = (odds - expected_avg_price) / odds # 0.02 / 2.0 = 0.01

    exec_price, slippage = engine.simulate_execution(stake, odds, league)
    assert math.isclose(exec_price, expected_avg_price, rel_tol=1e-9)
    assert math.isclose(slippage, expected_slippage, rel_tol=1e-9)

    # Stake consumes 10 ticks
    stake = depth_per_tick * 10.0 # 500.0
    ticks_consumed = 10.0
    expected_avg_price = odds - (ticks_consumed / 2.0) * tick_size # 2.0 - 5.0 * 0.01 = 1.95
    expected_slippage = (odds - expected_avg_price) / odds # 0.05 / 2.0 = 0.025

    exec_price, slippage = engine.simulate_execution(stake, odds, league)
    assert math.isclose(exec_price, expected_avg_price, rel_tol=1e-9)
    assert math.isclose(slippage, expected_slippage, rel_tol=1e-9)

def test_simulate_execution_capped_slippage(engine):
    """Test very large stakes where avg_price is capped at odds * 0.8."""
    odds = 2.0
    league = "Default"
    base_volume = engine.LEAGUE_LIQUIDITY[league] # 5000.0
    depth_per_tick = base_volume * 0.01 # 50.0
    tick_size = 0.01
    min_price = odds * 0.8 # 1.6

    # Stake consumes 100 ticks
    stake = depth_per_tick * 100.0 # 5000.0
    ticks_consumed = 100.0
    calculated_avg_price = odds - (ticks_consumed / 2.0) * tick_size # 2.0 - 50.0 * 0.01 = 1.5

    # 1.5 is less than min_price 1.6, so it should be capped
    expected_avg_price = min_price
    expected_slippage = (odds - expected_avg_price) / odds # 0.4 / 2.0 = 0.2

    exec_price, slippage = engine.simulate_execution(stake, odds, league)
    assert math.isclose(exec_price, expected_avg_price, rel_tol=1e-9)
    assert math.isclose(slippage, expected_slippage, rel_tol=1e-9)

def test_simulate_execution_league_routing(engine):
    """Test that different leagues yield different slippages, and unknown defaults correctly."""
    odds = 2.0
    stake = 200.0
    tick_size = 0.01

    # Premier League (100_000.0 base volume) -> depth_per_tick = 1000.0
    # Stake 200.0 consumes 0.2 ticks -> < 1.0 tick, no slippage
    exec_price, slippage = engine.simulate_execution(stake, odds, "Premier League")
    assert exec_price == odds
    assert slippage == 0.0

    # Championship (20_000.0 base volume) -> depth_per_tick = 200.0
    # Stake 200.0 consumes exactly 1.0 tick
    # expected_avg = 2.0 - (1.0 / 2.0) * 0.01 = 1.995
    exec_price, slippage = engine.simulate_execution(stake, odds, "Championship")
    expected_avg_price = odds - (1.0 / 2.0) * tick_size
    expected_slippage = (odds - expected_avg_price) / odds
    assert math.isclose(exec_price, expected_avg_price, rel_tol=1e-9)
    assert math.isclose(slippage, expected_slippage, rel_tol=1e-9)

    # Unknown league falls back to Default (5_000.0 base volume)
    # depth_per_tick = 50.0. Stake 200 consumes 4 ticks.
    # expected_avg = 2.0 - (4.0 / 2.0) * 0.01 = 1.98
    exec_price_unknown, slippage_unknown = engine.simulate_execution(stake, odds, "Some Random League")
    exec_price_default, slippage_default = engine.simulate_execution(stake, odds, "Default")

    assert math.isclose(exec_price_unknown, exec_price_default, rel_tol=1e-9)
    assert math.isclose(slippage_unknown, slippage_default, rel_tol=1e-9)
    assert math.isclose(exec_price_unknown, 1.98, rel_tol=1e-9)
