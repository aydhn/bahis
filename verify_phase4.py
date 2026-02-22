
import asyncio
from src.quant.monte_carlo_engine import MonteCarloEngine
from src.quant.elo_engine import EloEngine
from src.quant.philosophical_engine import PhilosophicalEngine, EpistemicReport

def test_monte_carlo():
    print("\n[TEST] Monte Carlo Simulation...")
    mc = MonteCarloEngine(simulations=1000, horizon=100)
    # Bankroll: 1000, WinRate: 0.55, Odds: 2.0, Stake: 0.05 (Kelly/2)
    res = mc.run(bankroll=1000.0, win_rate=0.55, avg_odds=2.0, stake_pct=0.05)
    
    print(f"Risk of Ruin: {res.risk_of_ruin:.1%}")
    print(f"Exp Final: {res.expected_final_bankroll:.2f}")
    
    if res.simulations_count == 1000 and res.expected_final_bankroll > 0:
        print("✅ Monte Carlo SUCCESS")
        return True
    return False

def test_elo():
    print("\n[TEST] Elo Engine...")
    elo = EloEngine()
    
    # Team A (1500) vs Team B (1500)
    # Home (A) wins
    elo.update_ratings("Team A", "Team B", 2, 0)
    
    rt_a = elo.get_rating("Team A")
    rt_b = elo.get_rating("Team B")
    
    print(f"Team A: {rt_a:.2f}, Team B: {rt_b:.2f}")
    
    if rt_a > 1500 and rt_b < 1500:
        print("✅ Elo Update SUCCESS")
        return True
    return False

def test_dialectic():
    print("\n[TEST] Dialectic Philosopher...")
    philo = PhilosophicalEngine()
    
    # Create a dummy report
    report = EpistemicReport(model_probability=0.7, epistemic_approved=True, epistemic_score=0.9)
    # This should trigger the new generate_commentary logic
    comment = philo.generate_commentary(report)
    
    print(f"Commentary: {comment}")
    
    if "Tez" in comment or "|" in comment or "=>" in comment or "[Diyalektik]" in comment:
        print("✅ Dialectic Commentary SUCCESS")
        return True
    return False

def main():
    mc = test_monte_carlo()
    elo = test_elo()
    dia = test_dialectic()
    
    if mc and elo and dia:
        print("\n🚀 ALL PHASE 4 CAPABILITIES VERIFIED")
    else:
        print("\n⚠️ SOME TESTS FAILED")

if __name__ == "__main__":
    main()
