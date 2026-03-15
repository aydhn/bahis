import unittest
from src.pipeline.stages.validator import DataValidatorStage

class TestDataValidatorStage(unittest.TestCase):
    def setUp(self):
        self.validator = DataValidatorStage()

    def test_missing_match_id(self):
        """Test that a missing match_id results in validation failure."""
        row = {
            "home_team": "Team A",
            "away_team": "Team B",
            "home_odds": 2.0,
            "draw_odds": 3.0,
            "away_odds": 4.0
        }
        is_valid, reason = self.validator._validate_match(row)
        self.assertFalse(is_valid)
        self.assertEqual(reason, "Missing match_id")

    def test_empty_row(self):
        """Test that an empty row fails gracefully with Missing match_id."""
        row = {}
        is_valid, reason = self.validator._validate_match(row)
        self.assertFalse(is_valid)
        self.assertEqual(reason, "Missing match_id")

if __name__ == '__main__':
    unittest.main()
