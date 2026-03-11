import pytest
from src.quant.finance.arbitrage_execution import ArbitrageExecutionManager

def test_arbitrage_execution():
    manager = ArbitrageExecutionManager()
    signal = {
        "type": "ARBITRAGE",
        "roi": 0.05,
        "home": {"odds": 2.5, "bookie": "A"},
        "draw": {"odds": 3.5, "bookie": "B"},
        "away": {"odds": 3.5, "bookie": "C"},
        "implied_prob": 1/2.5 + 1/3.5 + 1/3.5 # 0.4 + 0.285 + 0.285 = 0.97
    }
    plan = manager.plan_execution("test_match", signal, max_total_stake=10.0)
    assert plan.total_stake > 0
    assert len(plan.legs) == 3
