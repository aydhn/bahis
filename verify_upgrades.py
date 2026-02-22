from src.core.regime_kelly import RegimeKelly
from src.quant.philosophical_engine import PhilosophicalEngine

def test_kelly():
    print("Testing RegimeKelly kwargs support...")
    try:
        rk = RegimeKelly()
        # Passing extra arg 'context_data' to simulate Orchestrator behavior
        decision = rk.calculate(probability=0.6, odds=2.0, match_id="TEST_MATCH", context_data="dummy")
        print(f"Kelly Decision: {decision.stake_amount} (Approved: {decision.approved})")
        return True
    except TypeError as e:
        print(f"Kelly Failed: {e}")
        return False

def test_philosopher():
    print("\nTesting PhilosophicalEngine commentary...")
    try:
        pe = PhilosophicalEngine()
        report = pe.evaluate(probability=0.8, confidence=0.9, sample_size=300, match_id="TEST_PHILO")
        # Force approval
        report.epistemic_approved = True
        report.epistemic_score = 0.9
        
        comment = pe.generate_commentary(report)
        print(f"Commentary: {comment}")
        return True
    except Exception as e:
        print(f"Philosopher Failed: {e}")
        return False

if __name__ == "__main__":
    k = test_kelly()
    p = test_philosopher()
    if k and p:
        print("\n✅ Verification SUCCESS")
    else:
        print("\n❌ Verification FAILED")
