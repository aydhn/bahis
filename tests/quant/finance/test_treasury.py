import sys
from unittest.mock import MagicMock

# Mock out heavy dependencies due to restricted environment per memory instructions
sys.modules['scipy.stats'] = MagicMock()
sys.modules['scipy.optimize'] = MagicMock()
sys.modules['scipy.special'] = MagicMock()
sys.modules['polars'] = MagicMock()
sys.modules['numba'] = MagicMock()
sys.modules['torch'] = MagicMock()
sys.modules['torch.nn'] = MagicMock()
sys.modules['torch.optim'] = MagicMock()
sys.modules['torch.nn.functional'] = MagicMock()
sys.modules['cv2'] = MagicMock()

# Mock out config to avoid environment errors in test runner
class MockSettings:
    DATA_DIR = MagicMock()
sys.modules['pydantic_settings'] = MagicMock()
sys.modules['src.system.config'] = MagicMock()
sys.modules['src.system.config'].settings = MockSettings()


import pytest
from src.quant.finance.treasury import TreasuryEngine, TreasuryState

class TestTreasuryEngine:

    def setup_method(self):
        self.engine = TreasuryEngine()
        # Reset to known default state for tests
        self.engine.state = TreasuryState(
            total_capital=10000.0,
            daily_pnl=0.0,
            max_daily_drawdown_pct=0.05,
            buckets={"safe": 5000.0, "aggressive": 3000.0, "rnd": 2000.0},
            allocations={"safe": 0.5, "aggressive": 0.3, "rnd": 0.2},
            locked_capital=0.0
        )
        self.engine.save_state = MagicMock() # Prevent file IO during tests

    def test_get_sniper_stake_success(self):
        # 1% of 10000 is 100, but capped at 50
        stake = self.engine.get_sniper_stake()
        assert stake == 50.0

    def test_get_sniper_stake_small_capital(self):
        self.engine.state.total_capital = 1000.0
        # 1% of 1000 is 10, smaller than 50
        stake = self.engine.get_sniper_stake()
        assert stake == 10.0

    def test_get_sniper_stake_illiquid(self):
        self.engine.state.locked_capital = 9990.0 # Only 10 available
        stake = self.engine.get_sniper_stake()
        assert stake == 0.0

    def test_request_capital_success(self):
        approved = self.engine.request_capital(100.0, "safe")
        assert approved == 100.0
        assert self.engine.state.buckets["safe"] == 4900.0
        assert self.engine.state.locked_capital == 100.0

    def test_request_capital_circuit_breaker(self):
        self.engine.state.daily_pnl = -600.0 # > 5% of 10000
        approved = self.engine.request_capital(100.0, "safe")
        assert approved == 0.0

    def test_request_capital_empty_bucket(self):
        self.engine.state.buckets["safe"] = 0.0
        approved = self.engine.request_capital(100.0, "safe")
        assert approved == 0.0

    def test_request_capital_liquidity_limit(self):
        # 5% cash buffer required = 500.0
        # Available liquid = 10000 - 8500 = 1500
        # Max approvable = 1500 - 500 = 1000
        self.engine.state.locked_capital = 8500.0
        approved = self.engine.request_capital(1000.0, "safe")
        assert approved == 1000.0

    def test_release_capital_win(self):
        self.engine.state.locked_capital = 100.0
        self.engine.state.buckets["safe"] = 4900.0

        # Won bet, return stake (100) + profit (50)
        self.engine.release_capital(100.0, 50.0, "safe")

        assert self.engine.state.locked_capital == 0.0
        assert self.engine.state.buckets["safe"] == 5050.0
        assert self.engine.state.total_capital == 10050.0
        assert self.engine.state.daily_pnl == 50.0

    def test_release_capital_loss(self):
        self.engine.state.locked_capital = 100.0
        self.engine.state.buckets["safe"] = 4900.0

        # Lost bet, return stake (100) + loss (-100) -> 0 returned to bucket
        self.engine.release_capital(100.0, -100.0, "safe")

        assert self.engine.state.locked_capital == 0.0
        assert self.engine.state.buckets["safe"] == 4900.0
        assert self.engine.state.total_capital == 9900.0
        assert self.engine.state.daily_pnl == -100.0

    def test_rebalance_buckets_crash_regime(self):
        self.engine.rebalance_buckets("CRASH")
        assert self.engine.state.allocations["safe"] == 1.0
        assert self.engine.state.allocations["aggressive"] == 0.0
        assert self.engine.state.buckets["safe"] == 10000.0
        assert self.engine.state.buckets["aggressive"] == 0.0

    def test_stress_test_portfolio_solvent(self):
        self.engine.state.locked_capital = 5000.0
        res = self.engine.stress_test_portfolio(0.20) # 20% loss = 1000
        assert res["status"] == "SOLVENT"
        assert res["projected_loss"] == 1000.0
        assert res["projected_capital"] == 9000.0

    def test_stress_test_portfolio_illiquid(self):
        self.engine.state.total_capital = 10000.0
        self.engine.state.locked_capital = 9800.0 # Almost everything locked
        res = self.engine.stress_test_portfolio(0.20) # 20% loss = 1960
        # Projected capital = 10000 - 1960 = 8040
        # Remaining locked = 9800 - 1960 = 7840
        # Liquid = 8040 - 7840 = 200
        # Liquidity ratio = 200 / 8040 = 0.024 < 0.05
        assert res["status"] == "ILLIQUID"

    def test_reset_daily_pnl(self):
        self.engine.state.daily_pnl = 500.0
        self.engine.reset_daily_pnl()
        assert self.engine.state.daily_pnl == 0.0
