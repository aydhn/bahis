"""
test_stress_syndicate.py – Unit Tests for Advanced Risk & Consensus.
"""
import pytest
import numpy as np
from src.quant.finance.stress_tester import PortfolioStressTester
from src.quant.analysis.syndicate_consensus import SyndicateConsensus

# ---------------------------------------------------------
# Test Portfolio Stress Tester
# ---------------------------------------------------------
def test_stress_tester_healthy():
    tester = PortfolioStressTester(n_sims=500, var_threshold=0.20)

    current_bets = [
        {"odds": 2.0, "prob_home": 0.55, "stake_amount": 100},
        {"odds": 1.5, "prob_home": 0.70, "stake_amount": 100}
    ]
    new_bet = {"odds": 2.0, "prob_home": 0.55, "stake_amount": 100}

    # Total Capital 5000 -> 300 risked is low risk
    res = tester.check_portfolio_health(current_bets, new_bet, total_capital=5000.0)

    assert res["approved"] is True
    assert res["var_pct"] < 0.20
    print(f"\n[Stress Healthy] VaR: {res['var_pct']:.2%}")

def test_stress_tester_fail():
    tester = PortfolioStressTester(n_sims=500, var_threshold=0.10) # Strict 10% limit

    # Massive positions relative to bankroll
    current_bets = [
        {"odds": 2.0, "prob_home": 0.5, "stake_amount": 2000} # 40% of bankroll!
    ]
    new_bet = {"odds": 2.0, "prob_home": 0.5, "stake_amount": 2000} # Another 40%

    res = tester.check_portfolio_health(current_bets, new_bet, total_capital=5000.0)

    assert res["approved"] is False
    assert "Portfolio VaR Violation" in res["reason"]
    print(f"\n[Stress Fail] Reason: {res['reason']}")

# ---------------------------------------------------------
# Test Syndicate Consensus
# ---------------------------------------------------------
def test_syndicate_agreement():
    syndicate = SyndicateConsensus()

    outputs = {
        "benter": {"prob_home": 0.60},
        "lstm": {"prob_home": 0.62},
        "dixon": {"prob_home": 0.58}
    }

    verdict = syndicate.adjudicate(outputs)

    assert verdict.disagreement_level < 0.05
    assert "Strong Unanimous Consensus" in verdict.verdict_text
    assert 0.58 <= verdict.prob_home <= 0.62
    print(f"\n[Syndicate Agreement] Verdict: {verdict.verdict_text}")

def test_syndicate_conflict():
    syndicate = SyndicateConsensus()

    outputs = {
        "benter": {"prob_home": 0.80}, # High
        "lstm": {"prob_home": 0.20},   # Low (Conflict!)
        "dixon": {"prob_home": 0.50}
    }

    verdict = syndicate.adjudicate(outputs)

    assert verdict.disagreement_level > 0.15
    assert "High Conflict" in verdict.verdict_text
    # Confidence should be penalized
    assert verdict.confidence < 0.8 # Max is 0.8 but penalty applies
    print(f"\n[Syndicate Conflict] Verdict: {verdict.verdict_text}, Conf: {verdict.confidence:.2f}")

if __name__ == "__main__":
    test_stress_tester_healthy()
    test_stress_tester_fail()
    test_syndicate_agreement()
    test_syndicate_conflict()
