
from src.core.portfolio_optimizer import PortfolioOptimizer, PortfolioBet, DrawdownConfig, TradingMode
from src.quant.finance.liquidity_engine import LiquidityEngine

def test_optimizer():
    # Mock LiquidityEngine
    class MockLiquidity:
        def calculate_max_safe_stake(self, odds, edge, league):
            return 1000.0 # generous limit

    opt = PortfolioOptimizer(initial_bankroll=10000, liquidity_engine=MockLiquidity())

    bets = [
        PortfolioBet("m1", "HOME", 2.0, 0.55, 0.1, 0.05, "PL"),
        PortfolioBet("m2", "HOME", 2.0, 0.55, 0.1, 0.05, "PL"), # High correlation with m1 (same league)
        PortfolioBet("m3", "AWAY", 3.0, 0.4, 0.2, 0.05, "LaLiga")
    ]

    print("Testing STABLE regime...")
    res_stable = opt.optimize(bets, regime="STABLE")
    print(f"STABLE results: {len(res_stable)}")
    for r in res_stable:
        print(f"  {r['match_id']}: stake={r['adjusted_stake_pct']:.4f}")

    print("\nTesting CRASH regime...")
    res_crash = opt.optimize(bets, regime="CRASH")
    print(f"CRASH results: {len(res_crash)}")
    for r in res_crash:
        print(f"  {r['match_id']}: stake={r['adjusted_stake_pct']:.4f}")

if __name__ == "__main__":
    test_optimizer()
