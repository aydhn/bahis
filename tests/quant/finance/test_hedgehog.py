import unittest
import types
import sys
from unittest.mock import MagicMock, patch

class TestHedgeHog(unittest.TestCase):
    def setUp(self):
        # Create mock module objects to avoid MissingModule errors
        scipy_mock = types.ModuleType("scipy")
        scipy_stats = types.ModuleType("scipy.stats")
        scipy_stats.poisson = MagicMock()
        scipy_special = types.ModuleType("scipy.special")
        scipy_special.factorial = MagicMock()

        self.mock_modules = {
            'numpy': MagicMock(),
            'loguru': MagicMock(),
            'scipy': scipy_mock,
            'scipy.stats': scipy_stats,
            'scipy.special': scipy_special,
            'scipy.optimize': MagicMock(),
            'scipy.integrate': MagicMock(),
            'scipy.signal': MagicMock(),
            'scipy.interpolate': MagicMock(),
            'pandas': MagicMock(),
            'sklearn': MagicMock(),
            'sklearn.ensemble': MagicMock(),
            'numba': MagicMock(),
            'polars': MagicMock(),
            'torch': MagicMock(),
            'cv2': MagicMock(),
            'pydantic_settings': MagicMock(),
            'src.quant.finance.black_scholes_hedge': MagicMock(),
            'src.quant.models': MagicMock(),
            'src.quant.analysis': MagicMock(),
            'src.quant.physics': MagicMock(),
            'src.quant.risk': MagicMock(),
        }

        # Patch sys.modules safely for the duration of the test
        self.patcher = patch.dict('sys.modules', self.mock_modules)
        self.patcher.start()

        # Import the class under test AFTER mocking dependencies
        from src.quant.finance.hedgehog import HedgeHog
        self.hedgehog = HedgeHog()

    def tearDown(self):
        self.patcher.stop()
        # Explicitly remove the imported module to prevent test pollution
        if 'src.quant.finance.hedgehog' in sys.modules:
            del sys.modules['src.quant.finance.hedgehog']
        # if 'src.quant' in sys.modules:
            # del sys.modules['src.quant']

    def test_check_hedge_opportunity_invalid_odds(self):
        """Test with invalid original or live odds (<= 1.0)."""
        position = {"match_id": "1", "selection": "HOME", "stake": 100, "odds": 0.0}
        self.assertIsNone(self.hedgehog.check_hedge_opportunity(position, 2.0))

        position["odds"] = 2.0
        self.assertIsNone(self.hedgehog.check_hedge_opportunity(position, 0.0))

    def test_check_hedge_opportunity_profit_taking(self):
        """Test profit taking (Green Book) scenario when odds drop significantly."""
        position = {"match_id": "1", "selection": "HOME", "stake": 100, "odds": 3.0}
        # Odds drop by > 20% (3.0 -> 2.0)
        res = self.hedgehog.check_hedge_opportunity(position, 2.0)

        self.assertIsNotNone(res)
        self.assertEqual(res["action"], "HEDGE_PROFIT")
        self.assertIn("Odds dropped 3.0->2.0", res["reason"])
        self.assertIn("guaranteed_profit", res["details"])
        self.assertTrue(res["details"]["guaranteed_profit"] > 10) # Min 10% profit

    def test_check_hedge_opportunity_profit_taking_insufficient_profit(self):
        """Test profit taking scenario but profit is too small (< 10% stake)."""
        position = {"match_id": "1", "selection": "HOME", "stake": 100, "odds": 3.0}

        # calculate_green_book is mocked to return profit < 10 (10% of 100 stake)
        with patch.object(self.hedgehog, 'calculate_green_book', return_value={"guaranteed_profit": 5.0}):
            res = self.hedgehog.check_hedge_opportunity(position, 2.0)
            self.assertIsNone(res)

    def test_check_hedge_opportunity_stop_loss(self):
        """Test stop loss scenario when odds drift against us significantly."""
        position = {"match_id": "1", "selection": "HOME", "stake": 100, "odds": 2.0}
        # Odds drift by > 30% (2.0 -> 2.7)
        res = self.hedgehog.check_hedge_opportunity(position, 2.7)

        self.assertIsNotNone(res)
        self.assertEqual(res["action"], "STOP_LOSS")
        self.assertIn("Odds drifted 2.0->2.7", res["reason"])
        self.assertIn("loss", res)
        self.assertIn("cashout_value", res)

        # Cashout = (100 * 2.0) / 2.7 = 74.07
        # Loss = 100 - 74.07 = 25.93
        self.assertAlmostEqual(res["cashout_value"], 74.07, places=2)
        self.assertAlmostEqual(res["loss"], 25.93, places=2)

    def test_check_hedge_opportunity_hold(self):
        """Test hold scenario when odds drift is within tolerance (no action)."""
        position = {"match_id": "1", "selection": "HOME", "stake": 100, "odds": 2.0}

        # Odds drop slightly (not > 20%)
        res = self.hedgehog.check_hedge_opportunity(position, 1.8)
        self.assertIsNone(res)

        # Odds drift slightly (not > 30%)
        res = self.hedgehog.check_hedge_opportunity(position, 2.4)
        self.assertIsNone(res)

if __name__ == '__main__':
    unittest.main()
