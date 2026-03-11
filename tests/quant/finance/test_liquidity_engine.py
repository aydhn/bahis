import sys
import math
from unittest.mock import MagicMock

# Mock out heavy dependencies due to restricted environment per memory instructions
sys.modules['scipy.stats'] = MagicMock()
sys.modules['scipy.optimize'] = MagicMock()
sys.modules['scipy.special'] = MagicMock()

sys.modules['numba'] = MagicMock()
sys.modules['torch'] = MagicMock()
sys.modules['torch.nn'] = MagicMock()
sys.modules['torch.optim'] = MagicMock()
sys.modules['torch.nn.functional'] = MagicMock()
sys.modules['cv2'] = MagicMock()
sys.modules['pydantic_settings'] = MagicMock()


import pytest
from src.quant.finance.liquidity_engine import LiquidityEngine

class TestLiquidityEngine:

    def setup_method(self):
        self.engine = LiquidityEngine()

    def test_simulate_execution_small_stake(self):
        # 100 units is tiny, shouldn't move odds
        exec_price, slippage = self.engine.simulate_execution(100.0, 2.0, "Premier League")
        assert exec_price == 2.0
        assert slippage == 0.0

    def test_simulate_execution_large_stake(self):
        # Base vol for Premier is 100k. Depth per tick = 1000.
        # Stake 5000 means 5 ticks consumed.
        # Avg price = 2.0 - (5/2)*0.01 = 2.0 - 0.025 = 1.975
        exec_price, slippage = self.engine.simulate_execution(5000.0, 2.0, "Premier League")
        assert math.isclose(exec_price, 1.975, rel_tol=1e-5)
        assert slippage > 0.0

    def test_simulate_execution_illiquid_league(self):
        # Base vol for Default is 5k. Depth per tick = 50.
        # Stake 500 means 10 ticks.
        # Avg price = 2.0 - (10/2)*0.01 = 2.0 - 0.05 = 1.95
        exec_price, slippage = self.engine.simulate_execution(500.0, 2.0, "Unknown League")
        assert math.isclose(exec_price, 1.95, rel_tol=1e-5)

    def test_calculate_max_safe_stake(self):
        # Edge = 0.05. Target slippage = 0.025
        # Ticks allowed = (0.025 * 2.0 * 2) / 0.01 = 10 ticks
        # Depth per tick for Premier League = 1000.
        # Max stake = 10 * 1000 = 10000
        max_stake = self.engine.calculate_max_safe_stake(odds=2.0, edge=0.05, league="Premier League")
        assert max_stake == 10000.0

    def test_estimate_impact(self):
        # Premier League vol = 100k
        assert self.engine.estimate_impact(50, "Premier League") == "Invisible"
        assert self.engine.estimate_impact(500, "Premier League") == "Low Impact"
        assert self.engine.estimate_impact(3000, "Premier League") == "Moderate Slip"
        assert self.engine.estimate_impact(10000, "Premier League") == "Market Mover (High Slip)"
