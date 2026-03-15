from unittest.mock import MagicMock
import pytest
import numpy as np

from src.core.portfolio_optimizer import (
    PortfolioOptimizer,
    PortfolioBet,
    TradingMode,
    DrawdownConfig
)

class MockLiquidityEngine:
    def calculate_max_safe_stake(self, odds: float, edge: float, league: str) -> float:
        return 1000.0 # 10% of 10k initial bankroll

def test_initialization():
    optimizer = PortfolioOptimizer(initial_bankroll=10000.0, max_portfolio_risk=0.15)
    assert optimizer.bankroll == 10000.0
    assert optimizer.mode == TradingMode.LIVE
    assert optimizer.drawdown == 0.0

def test_regime_risk_factors():
    optimizer = PortfolioOptimizer()
    assert optimizer._get_regime_risk_factor("STABLE") == 1.0
    assert optimizer._get_regime_risk_factor("VOLATILE") == 0.6
    assert optimizer._get_regime_risk_factor("CHAOTIC") == 0.3
    assert optimizer._get_regime_risk_factor("CRASH") == 0.1
    assert optimizer._get_regime_risk_factor("UNKNOWN") == 1.0

    assert optimizer._get_risk_aversion("STABLE") == 1.5
    assert optimizer._get_risk_aversion("VOLATILE") == 3.0
    assert optimizer._get_risk_aversion("CHAOTIC") == 5.0
    assert optimizer._get_risk_aversion("CRASH") == 10.0
    assert optimizer._get_risk_aversion("UNKNOWN") == 2.0

def test_correlation_estimation():
    optimizer = PortfolioOptimizer()
    bet1 = PortfolioBet(match_id="m1", selection="home", odds=2.0, prob_model=0.55, ev=0.1, stake_pct=0.05, league="EPL", correlation_group="g1")
    bet2 = PortfolioBet(match_id="m1", selection="over", odds=1.9, prob_model=0.55, ev=0.05, stake_pct=0.02, league="EPL", correlation_group="g2")
    bet3 = PortfolioBet(match_id="m2", selection="home", odds=2.0, prob_model=0.55, ev=0.1, stake_pct=0.05, league="EPL", correlation_group="g1")
    bet4 = PortfolioBet(match_id="m3", selection="away", odds=3.0, prob_model=0.35, ev=0.05, stake_pct=0.02, league="LaLiga", correlation_group="g3")

    # Same match
    assert optimizer._estimate_correlation(bet1, bet2) == 0.7

    # Same league (0.15), same selection (0.1), same group (0.2) = 0.45
    assert optimizer._estimate_correlation(bet1, bet3) == 0.45

    # Different everything
    assert optimizer._estimate_correlation(bet1, bet4) == 0.0

def test_correlation_matrix():
    optimizer = PortfolioOptimizer()
    bets = [
        PortfolioBet(match_id="m1", selection="home", odds=2.0, prob_model=0.55, ev=0.1, stake_pct=0.05, league="EPL"),
        PortfolioBet(match_id="m1", selection="away", odds=3.0, prob_model=0.35, ev=0.05, stake_pct=0.02, league="EPL")
    ]
    corr = optimizer._build_correlation_matrix(bets)
    assert corr.shape == (2, 2)
    assert corr[0, 0] == 1.0
    assert corr[1, 1] == 1.0
    assert corr[0, 1] == 0.7
    assert corr[1, 0] == 0.7

def test_drawdown_control():
    config = DrawdownConfig(reduce_threshold=0.1, paper_threshold=0.2, freeze_threshold=0.3, recovery_threshold=0.05, cooldown_hours=0)
    optimizer = PortfolioOptimizer(initial_bankroll=1000.0, drawdown_config=config)

    # Peak is 1000
    optimizer.update_bankroll(-100) # Bankroll 900, DD 10%
    assert optimizer.mode == TradingMode.REDUCED
    assert optimizer._mode_multiplier() == 0.5

    optimizer.update_bankroll(-100) # Bankroll 800, DD 20%
    assert optimizer.mode == TradingMode.PAPER
    assert optimizer._mode_multiplier() == 1.0

    optimizer.update_bankroll(-150) # Bankroll 650, DD 35%
    assert optimizer.mode == TradingMode.FROZEN
    assert optimizer._mode_multiplier() == 0.0

    # Recover to 950, DD 5%
    optimizer.update_bankroll(300)
    assert optimizer.mode == TradingMode.FROZEN


def test_optimize_empty_and_negative_ev():
    optimizer = PortfolioOptimizer()
    assert optimizer.optimize([]) == []

    bets = [PortfolioBet(match_id="m1", selection="home", odds=2.0, prob_model=0.4, ev=-0.2, stake_pct=0.0)]
    assert optimizer.optimize(bets) == []

def test_optimize_basic():
    optimizer = PortfolioOptimizer(initial_bankroll=10000.0, max_portfolio_risk=0.10, liquidity_engine=MockLiquidityEngine())
    bets = [
        PortfolioBet(match_id="m1", selection="home", odds=2.0, prob_model=0.6, ev=0.2, stake_pct=0.05, league="EPL"),
        PortfolioBet(match_id="m2", selection="away", odds=3.0, prob_model=0.4, ev=0.2, stake_pct=0.05, league="LaLiga")
    ]

    results = optimizer.optimize(bets)
    assert len(results) == 2

    total_risk = sum(r["adjusted_stake_pct"] for r in results)
    assert total_risk <= 0.10 # Max risk check
    assert results[0]["trading_mode"] == "LIVE"
    assert not results[0]["is_paper"]

def test_optimize_frozen_mode():
    config = DrawdownConfig(freeze_threshold=0.1)
    optimizer = PortfolioOptimizer(initial_bankroll=1000.0, drawdown_config=config)
    optimizer.update_bankroll(-200) # Freeze

    bets = [PortfolioBet(match_id="m1", selection="home", odds=2.0, prob_model=0.6, ev=0.2, stake_pct=0.05)]
    results = optimizer.optimize(bets)
    assert results == []



def test_optimize_crash_regime():
    from src.system.container import container
    from unittest.mock import patch

    optimizer = PortfolioOptimizer(initial_bankroll=10000.0, max_portfolio_risk=0.10, liquidity_engine=MockLiquidityEngine())

    bets = [
        PortfolioBet(match_id="m1", selection="home", odds=2.0, prob_model=0.6, ev=0.2, stake_pct=0.05, league="EPL")
    ]

    kelly_benter = container.get('kelly_benter')

    if kelly_benter is not None:
        with patch.object(kelly_benter, 'calculate_fraction', return_value=0.05):
            results = optimizer.optimize(bets, regime="CRASH")
    else:
        results = optimizer.optimize(bets, regime="CRASH")

    assert len(results) == 1
    assert results[0]["adjusted_stake_pct"] <= 0.0101
def test_eigen_risk_parity_fallback():
    optimizer = PortfolioOptimizer(initial_bankroll=10000.0, max_portfolio_risk=0.10, liquidity_engine=MockLiquidityEngine())

    bets = [
        PortfolioBet(match_id="m1", selection="home", odds=2.0, prob_model=0.6, ev=0.2, stake_pct=0.05, league="EPL"),
        PortfolioBet(match_id="m1", selection="over", odds=1.9, prob_model=0.55, ev=0.1, stake_pct=0.05, league="EPL"),
        PortfolioBet(match_id="m1", selection="btts", odds=1.8, prob_model=0.58, ev=0.15, stake_pct=0.05, league="EPL")
    ]

    corr = optimizer._build_correlation_matrix(bets)
    stakes = np.array([b.stake_pct for b in bets])
    mask = np.ones(len(bets), dtype=bool)

    adjusted = optimizer._eigen_risk_parity(stakes, corr, mask)
    assert np.sum(adjusted) <= np.sum(stakes)
