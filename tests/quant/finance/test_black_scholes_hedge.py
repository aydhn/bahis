import math
import unittest
from unittest.mock import patch
from src.quant.finance.black_scholes_hedge import BlackScholesHedge

def norm_cdf(x):
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

class TestBlackScholesHedge(unittest.TestCase):
    def setUp(self):
        self.hedge = BlackScholesHedge()

    def test_calculate_binary_call_happy_path(self):
        """Test binary call calculation with typical values."""
        S = 0.5
        K = 1.0
        T = 0.5
        sigma = 0.2
        r = 0.0

        d2 = (math.log(S / K) + (r - 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        expected_result = math.exp(-r * T) * norm_cdf(d2)

        result = self.hedge.calculate_binary_call(S, K, T, sigma, r)
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_calculate_binary_call_time_expiry(self):
        """Test binary call calculation when time <= 0."""
        self.assertEqual(self.hedge.calculate_binary_call(S=1.0, K=0.5, T=0.0, sigma=0.2), 1.0)
        self.assertEqual(self.hedge.calculate_binary_call(S=0.5, K=1.0, T=0.0, sigma=0.2), 0.0)
        self.assertEqual(self.hedge.calculate_binary_call(S=0.5, K=0.5, T=-1.0, sigma=0.2), 1.0)

    def test_calculate_binary_call_zero_sigma(self):
        """Test binary call calculation when sigma <= 0."""
        self.assertEqual(self.hedge.calculate_binary_call(S=1.0, K=0.5, T=1.0, sigma=0.0), 1.0)
        self.assertEqual(self.hedge.calculate_binary_call(S=0.5, K=1.0, T=1.0, sigma=-1.0), 0.0)

    def test_calculate_binary_call_boundary_zero(self):
        """Test boundary clamping where S=0 or K=0 to avoid log(0) domain errors."""
        T = 1.0
        sigma = 0.2
        r = 0.0

        result_s_zero = self.hedge.calculate_binary_call(S=0.0, K=1.0, T=T, sigma=sigma, r=r)
        self.assertIsInstance(result_s_zero, float)
        self.assertTrue(0.0 <= result_s_zero <= 1.0)

        result_k_zero = self.hedge.calculate_binary_call(S=1.0, K=0.0, T=T, sigma=sigma, r=r)
        self.assertIsInstance(result_k_zero, float)
        self.assertTrue(0.0 <= result_k_zero <= 1.0)

if __name__ == '__main__':
    unittest.main()
