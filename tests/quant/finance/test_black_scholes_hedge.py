import math
import types
import unittest
from unittest.mock import MagicMock, patch

# Define a real mock for norm.cdf to replace scipy.stats.norm.cdf
def norm_cdf(x):
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

class MockNorm:
    @staticmethod
    def cdf(x):
        return norm_cdf(x)

class TestBlackScholesHedge(unittest.TestCase):
    def setUp(self):
        # Create mock module objects
        scipy_mock = types.ModuleType("scipy")
        scipy_stats = types.ModuleType("scipy.stats")
        scipy_stats.norm = MockNorm()

        # Dictionary of modules to mock
        self.mock_modules = {
            'numpy': MagicMock(),
            'loguru': MagicMock(),
            'numba': MagicMock(),
            'polars': MagicMock(),
            'torch': MagicMock(),
            'cv2': MagicMock(),
            'pydantic_settings': MagicMock(),
            'scipy': scipy_mock,
            'scipy.stats': scipy_stats,
            'scipy.optimize': MagicMock(),
            'scipy.special': MagicMock(),
            'scipy.integrate': MagicMock(),
            'scipy.signal': MagicMock(),
            'scipy.interpolate': MagicMock(),
            'pandas': MagicMock(),
            'sklearn': MagicMock(),
            'sklearn.ensemble': MagicMock(),
            'src.quant.models': MagicMock(),
            'src.quant.analysis': MagicMock(),
            'src.quant.physics': MagicMock(),
            'src.quant.risk': MagicMock(),
        }

        # Patch sys.modules safely for the duration of the test
        self.patcher = patch.dict('sys.modules', self.mock_modules)
        self.patcher.start()

        # Import the class under test AFTER mocking dependencies
        from src.quant.finance.black_scholes_hedge import BlackScholesHedge
        self.hedge = BlackScholesHedge()

    def tearDown(self):
        self.patcher.stop()

    def test_calculate_binary_call_happy_path(self):
        """Test binary call calculation with typical values."""
        S = 0.5
        K = 1.0
        T = 0.5
        sigma = 0.2
        r = 0.0

        # External calculation of expected result
        d2 = (math.log(S / K) + (r - 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        expected_result = math.exp(-r * T) * norm_cdf(d2)

        result = self.hedge.calculate_binary_call(S, K, T, sigma, r)
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_calculate_binary_call_time_expiry(self):
        """Test binary call calculation when time <= 0."""
        # T=0, S >= K -> 1.0
        self.assertEqual(self.hedge.calculate_binary_call(S=1.0, K=0.5, T=0.0, sigma=0.2), 1.0)
        # T=0, S < K -> 0.0
        self.assertEqual(self.hedge.calculate_binary_call(S=0.5, K=1.0, T=0.0, sigma=0.2), 0.0)
        # T < 0, S >= K -> 1.0
        self.assertEqual(self.hedge.calculate_binary_call(S=0.5, K=0.5, T=-1.0, sigma=0.2), 1.0)

    def test_calculate_binary_call_zero_sigma(self):
        """Test binary call calculation when sigma <= 0."""
        # sigma=0, S >= K -> 1.0
        self.assertEqual(self.hedge.calculate_binary_call(S=1.0, K=0.5, T=1.0, sigma=0.0), 1.0)
        # sigma < 0, S < K -> 0.0
        self.assertEqual(self.hedge.calculate_binary_call(S=0.5, K=1.0, T=1.0, sigma=-1.0), 0.0)

    def test_calculate_binary_call_boundary_zero(self):
        """Test boundary clamping where S=0 or K=0 to avoid log(0) domain errors."""
        T = 1.0
        sigma = 0.2
        r = 0.0

        # S=0.0
        result_s_zero = self.hedge.calculate_binary_call(S=0.0, K=1.0, T=T, sigma=sigma, r=r)
        self.assertIsInstance(result_s_zero, float)
        self.assertTrue(0.0 <= result_s_zero <= 1.0)

        # K=0.0
        result_k_zero = self.hedge.calculate_binary_call(S=1.0, K=0.0, T=T, sigma=sigma, r=r)
        self.assertIsInstance(result_k_zero, float)
        self.assertTrue(0.0 <= result_k_zero <= 1.0)

if __name__ == '__main__':
    unittest.main()
