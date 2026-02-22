
import asyncio
import os
from src.memory.vector_memory import VectorMemory
from src.quant.hybrid_model import HybridGoalModel
from src.utils.visual_reporter import VisualReporter

def test_vector_memory():
    print("\n[TEST] Vector Memory...")
    vm = VectorMemory("data/test_vectors.json")
    
    # 1. Add Data
    vm.add_match("M1", {"home_xg": 2.5, "away_xg": 0.5}) # Dominant Home
    vm.add_match("M2", {"home_xg": 0.5, "away_xg": 2.5}) # Dominant Away
    vm.add_match("M3", {"home_xg": 2.4, "away_xg": 0.6}) # Similar to M1
    
    # 2. Search
    query = {"home_xg": 2.6, "away_xg": 0.4} # Should match M1 and M3
    results = vm.find_similar(query, top_k=2)
    
    print("Search Results:", results)
    
    ids = [r["match_id"] for r in results]
    if "M1" in ids and "M3" in ids:
        print("✅ Vector Search SUCCESS")
        return True
    return False

def test_hybrid_model():
    print("\n[TEST] Hybrid Goal Model (Weibull)...")
    model = HybridGoalModel(weibull_shape=1.5) # Increasing hazard
    
    # Late goal prob should be higher than early goal prob for same duration if XG is distributed
    # But wait, predict_late_goal_prob calculates for REMAINING time.
    
    # Let's check simulate_match_timeline
    timeline = model.simulate_match_timeline(home_xg=1.5, away_xg=1.0)
    
    first_half_prob = timeline[0].prob_goal_home # 0-10 min
    last_half_prob = timeline[-1].prob_goal_home # 80-90 min
    
    print(f"Prob (0-10m): {first_half_prob:.4f}")
    print(f"Prob (80-90m): {last_half_prob:.4f}")
    
    if last_half_prob > first_half_prob:
        print("✅ Hybrid Model (Late Goal Bias) SUCCESS")
        return True
    return False

def test_visual_reporter():
    print("\n[TEST] Visual Reporter...")
    vr = VisualReporter()
    
    img_bytes = vr.generate_pnl_chart([100, 110, 105, 120], ["1", "2", "3", "4"])
    
    if len(img_bytes) > 1000: # Valid image
        print(f"✅ Chart Generated ({len(img_bytes)} bytes)")
        
        # Save for manual inspection if needed
        # with open("test_chart.png", "wb") as f:
        #     f.write(img_bytes)
        return True
    return False

def main():
    v = test_vector_memory()
    h = test_hybrid_model()
    vis = test_visual_reporter()
    
    if v and h and vis:
        print("\n🚀 ALL PHASE 5 CAPABILITIES VERIFIED")
    else:
        print("\n⚠️ SOME TESTS FAILED")
        
    # Clean up test file
    if os.path.exists("data/test_vectors.json"):
        os.remove("data/test_vectors.json")

if __name__ == "__main__":
    main()
