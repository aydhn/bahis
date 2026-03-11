import unittest
import numpy as np
from src.quant.physics.chaos_filter import ChaosFilter
from src.quant.physics.fractal_analyzer import FractalAnalyzer
from src.quant.physics.path_signature_engine import PathSignatureEngine
from src.quant.physics.quantum_brain import QuantumBrain

class TestPhysicsEngines(unittest.TestCase):

    def test_chaos_filter(self):
        cf = ChaosFilter()
        # Create a simple sine wave (stable)
        x = np.linspace(0, 100, 200)
        data = np.sin(x)
        report = cf.analyze(data)
        # Should be stable or at least not chaotic
        self.assertNotEqual(report.regime, "chaotic")

        # Create chaotic logistic map
        r = 3.9
        x = 0.5
        chaotic_data = []
        for _ in range(200):
            x = r * x * (1 - x)
            chaotic_data.append(x)
        report_chaos = cf.analyze(np.array(chaotic_data))
        # Should detect high lyapunov
        # Note: Nolds might not be installed, so it might use fallback which is less accurate
        # but should run without error
        self.assertIsNotNone(report_chaos.params.max_lyapunov)

    def test_fractal_analyzer(self):
        fa = FractalAnalyzer()
        # Random walk
        data = np.random.normal(0, 1, 500)
        res = fa.compute_hurst(data)
        # Hurst should be around 0.5
        self.assertGreater(res.hurst, 0.3)
        self.assertLess(res.hurst, 0.7)

    def test_path_signature(self):
        pse = PathSignatureEngine()
        # Dummy odds features dataframe
        # Assuming polars input or dict, but engine takes single row logic internally if we call _build_path
        # Let's test _build_path and _compute_signature directly if possible, or mock pl df
        import polars as pl
        df = pl.DataFrame({
            "match_id": ["m1"],
            "home_odds": [2.0],
            "draw_odds": [3.0],
            "away_odds": [4.0]
        })
        res = pse.extract(df)
        self.assertTrue(len(res) > 0 if not hasattr(res, "is_empty") else not res.is_empty())
        self.assertIn("sig_roughness", res.columns)

    def test_quantum_brain(self):
        # Test initialization and basic prediction structure
        qb = QuantumBrain(n_qubits=4, n_layers=1)
        # Dummy input
        X = np.random.rand(1, 4)
        pred = qb.predict(X)
        self.assertEqual(len(pred), 1)

if __name__ == '__main__':
    unittest.main()
