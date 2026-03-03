import numpy as np
from src.quant.finance.stress_tester import PortfolioStressTester

def test_check_portfolio_health_bankrupt():
    tester = PortfolioStressTester()
    res = tester.check_portfolio_health([], {}, 0.0)
    assert res["approved"] is False
    assert "Bankrupt" in res["reason"]

def test_check_portfolio_health_healthy():
    tester = PortfolioStressTester(n_sims=1000, var_threshold=0.20)

    # Very safe bet: 1% stake
    new_bet = {"odds": 2.0, "prob_home": 0.5, "stake_amount": 10.0}
    res = tester.check_portfolio_health([], new_bet, 1000.0)

    assert res["approved"] is True
    assert "Healthy" in res["reason"]
    # 10 is 1% of 1000. Even losing it completely is 1% var.
    assert res["var_pct"] <= 0.01

def test_check_portfolio_health_violation():
    tester = PortfolioStressTester(n_sims=2000, var_threshold=0.20)

    # Very risky bet: 50% of bankroll
    new_bet = {"odds": 2.0, "prob_home": 0.5, "stake_amount": 500.0}
    res = tester.check_portfolio_health([], new_bet, 1000.0)

    assert res["approved"] is False
    assert "Violation" in res["reason"]
    assert res["var_pct"] > 0.20
    assert "stress_impact" in res

def test_check_portfolio_health_multiple_bets():
    tester = PortfolioStressTester(n_sims=5000, var_threshold=0.20)

    current_bets = [
        {"odds": 2.0, "prob_home": 0.5, "stake_pct": 0.1}, # 10%
        {"odds": 3.0, "prob_home": 0.33, "stake_pct": 0.05} # 5%
    ]
    new_bet = {"odds": 1.5, "prob_home": 0.66, "stake_amount": 150.0} # 15% on 1000

    res = tester.check_portfolio_health(current_bets, new_bet, 1000.0)

    # 10% + 5% + 15% = 30% of bankroll exposed.
    assert res["approved"] is False
    assert res["var_pct"] >= 0.25

def test_check_portfolio_health_fallback_values():
    tester = PortfolioStressTester(n_sims=1000, var_threshold=0.20)

    # Missing probs and stakes - should fallback safely
    new_bet = {}
    # Fallback uses odds=2.0, prob=0.5, stake=0.0
    res = tester.check_portfolio_health([], new_bet, 1000.0)

    assert res["approved"] is True
    assert res["var_pct"] == 0.0

def test_check_portfolio_health_zero_odds_fallback():
    tester = PortfolioStressTester(n_sims=1000, var_threshold=0.20)

    # Odds zero should result in probability 0.5
    new_bet = {"odds": 0.0, "stake_amount": 100.0}
    res = tester.check_portfolio_health([], new_bet, 1000.0)

    assert res["approved"] is True
    # If odds=0.0, probability=0.5. Stake=100.
    # Outcome will be losing 100 (since odds-1 is -1, winning gives -100). Wait:
    # `stakes * (odds - 1)`
    # if odds is 0.0, (odds - 1) = -1.
    # Win: 100 * -1 = -100 PNL. Loss: -100 PNL.
    # PNL is always -100.
    # var_pct = 100 / 1000 = 0.1
    # 0.1 <= 0.20, so approved.
    np.testing.assert_approx_equal(res["var_pct"], 0.1, significant=2)
