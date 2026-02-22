from src.quant.market_cycles import MarketCycles
from src.quant.pnl_tracker import PnLTracker, Transaction
import numpy as np

def test_fft():
    print("Testing FFT Market Cycles...")
    # Create a synthetic signal with a period of 4
    # e.g., Win, Loss, Win, Loss pattern repeated
    # [1, 0, 0, 0, 1, 0, 0, 0] -> Period 4
    data = [1, 0, 0, 0] * 10
    mc = MarketCycles(min_period=2.0)
    cycles = mc.analyze_series(data)
    
    if cycles and abs(cycles[0].period - 4.0) < 0.5:
        print(f"✅ FFT Cycle Detected: {cycles[0].description}")
        return True
    else:
        print(f"❌ FFT Failed. Found: {cycles}")
        return False

def test_pnl():
    print("\nTesting PnL Tracker...")
    tracker = PnLTracker("data/test_ledger.csv")
    
    t = Transaction(
        match_id="TEST_MATCH_001",
        selection="1",
        odds=2.0,
        stake=100.0,
        result="PENDING"
    )
    tracker.record_bet(t)
    
    tracker.update_result("TEST_MATCH_001", "WON")
    
    stats = tracker.get_stats()
    print(f"Stats: {stats}")
    
    if stats["pnl"] == 100.0 and stats["roi"] == 1.0:
        print("✅ PnL Calculation Correct")
        return True
    else:
        print("❌ PnL Calculation Failed")
        return False

if __name__ == "__main__":
    fft_ok = test_fft()
    pnl_ok = test_pnl()
    
    if fft_ok and pnl_ok:
        print("\n✅ Analytics Verification SUCCESS")
    else:
        print("\n❌ Analytics Verification FAILED")
