import unittest
from unittest.mock import MagicMock
from src.quant.finance.optimal_execution import OptimalExecutionModel, ExecutionSchedule

class TestOptimalExecutionModel(unittest.TestCase):
    def setUp(self):
        # We test using the actual implementations of numpy and scipy
        # as imported by OptimalExecutionModel
        self.model = OptimalExecutionModel(default_steps=5)

    def test_calculate_slicing_schedule_sum(self):
        """Test that the sum of the calculated slices equals exactly the total_stake."""
        total_stake = 1234.56
        schedule = self.model.calculate_slicing_schedule(
            total_stake=total_stake,
            urgency=0.5,
            volatility=0.02,
            base_liquidity=50000.0
        )

        calculated_sum = sum(slice_obj.stake_amount for slice_obj in schedule.slices)
        self.assertAlmostEqual(calculated_sum, total_stake, places=2)
        self.assertEqual(schedule.total_stake, total_stake)

    def test_calculate_slicing_schedule_urgency(self):
        """Test behavior at boundaries of urgency."""
        # High urgency should execute in 1 step
        schedule_high = self.model.calculate_slicing_schedule(
            total_stake=1000.0,
            urgency=0.95,
            volatility=0.02
        )
        self.assertEqual(len(schedule_high.slices), 1)
        self.assertEqual(schedule_high.duration_steps, 1)
        self.assertEqual(schedule_high.slices[0].stake_amount, 1000.0)

        # Low urgency should execute in multiple steps
        schedule_low = self.model.calculate_slicing_schedule(
            total_stake=1000.0,
            urgency=0.1,
            volatility=0.02
        )
        self.assertGreater(len(schedule_low.slices), 1)
        self.assertGreater(schedule_low.duration_steps, 1)

        calculated_sum_low = sum(slice_obj.stake_amount for slice_obj in schedule_low.slices)
        self.assertAlmostEqual(calculated_sum_low, 1000.0, places=2)

    def test_calculate_slicing_schedule_twap(self):
        """Test TWAP execution when kappa < 1e-4."""
        # kappa = urgency * volatility * 100.
        # So 0.1 * 0.000001 * 100 = 0.00001 < 1e-4
        schedule_twap = self.model.calculate_slicing_schedule(
            total_stake=1000.0,
            urgency=0.1,
            volatility=0.000001
        )
        self.assertGreater(len(schedule_twap.slices), 1)

        calculated_sum_twap = sum(slice_obj.stake_amount for slice_obj in schedule_twap.slices)
        self.assertAlmostEqual(calculated_sum_twap, 1000.0, places=2)

    def test_calculate_slicing_schedule_edge_cases(self):
        """Test edge cases like extremely low total stake compared to liquidity."""
        # Total stake < base_liquidity * 0.01 triggers 1 step
        schedule_tiny = self.model.calculate_slicing_schedule(
            total_stake=10.0,
            urgency=0.5,
            volatility=0.02,
            base_liquidity=50000.0
        )
        self.assertEqual(len(schedule_tiny.slices), 1)
        self.assertEqual(schedule_tiny.duration_steps, 1)
        self.assertEqual(schedule_tiny.slices[0].stake_amount, 10.0)

if __name__ == '__main__':
    unittest.main()
