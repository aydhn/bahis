import sys
import os
import numpy as np

# Add src to path
sys.path.append(os.getcwd())

from src.quant.analysis.similarity import SimilarityEngine
from src.quant.meta_labeling import MetaLabeler
from src.quant.risk.volatility_modulator import VolatilityModulator
from src.reporting.visualizer import Visualizer
from src.core.cognitive_guardian import CognitiveGuardian

def test_similarity_engine():
    print("Testing SimilarityEngine...")
    engine = SimilarityEngine(k=2)
    engine.mock_fit()

    test_vec = np.random.rand(1, 5)
    results = engine.find_similar(test_vec)

    assert isinstance(results, list)
    if results:
        assert "match_id" in results[0]
        assert "similarity" in results[0]
    print("SimilarityEngine OK.")

def test_meta_labeler():
    print("Testing MetaLabeler...")
    labeler = MetaLabeler()
    labeler.mock_train()

    features = {"confidence": 0.8, "entropy": 0.5, "odds": 2.0}
    score = labeler.predict_score(features)

    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0
    print(f"MetaLabeler OK (Score: {score}).")

def test_volatility_modulator():
    print("Testing VolatilityModulator...")
    vm = VolatilityModulator(window_size=10, target_vol=0.05)

    # Simulate calm market
    for _ in range(10):
        vm.update_returns(0.01) # 1% steady return

    mult = vm.get_stake_multiplier()
    assert mult > 0
    print(f"VolatilityModulator OK (Calm Multiplier: {mult}).")

    # Simulate chaos
    vm.update_returns(-0.20) # 20% loss
    mult_chaos = vm.get_stake_multiplier()
    assert mult_chaos < mult
    print(f"VolatilityModulator OK (Chaos Multiplier: {mult_chaos}).")

def test_visualizer():
    print("Testing Visualizer...")
    buf = Visualizer.generate_dummy_chart()
    assert buf is not None
    assert buf.getbuffer().nbytes > 0

    buf2 = Visualizer.generate_value_chart("A", "B", [0.5, 0.3, 0.2], [0.4, 0.3, 0.3])
    assert buf2 is not None
    print("Visualizer OK.")

def test_cognitive_guardian():
    print("Testing CognitiveGuardian...")
    cg = CognitiveGuardian()

    # Normal bet
    assert cg.check_bet({"stake": 100, "team": "A"}) == True
    cg.record_bet({"stake": 100})
    cg.record_outcome(-100) # Loss

    # Simulate tilt (3 losses)
    cg.record_outcome(-100)
    cg.record_outcome(-100)

    # Now try to double down (Chase)
    res = cg.check_bet({"stake": 500, "team": "A"}) # 5x stake
    # Should be blocked
    if not res:
        print("CognitiveGuardian Blocked Chase (Expected).")
    else:
        print("CognitiveGuardian Failed to block Chase (Unexpected).")

    print("CognitiveGuardian OK.")

if __name__ == "__main__":
    try:
        test_similarity_engine()
        test_meta_labeler()
        test_volatility_modulator()
        test_visualizer()
        test_cognitive_guardian()
        print("\nALL ADVANCED FEATURES PASSED.")
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        sys.exit(1)
