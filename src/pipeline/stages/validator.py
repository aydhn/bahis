from typing import Dict, Any, List
import polars as pl
from loguru import logger
from src.pipeline.core import PipelineStage

class DataValidatorStage(PipelineStage):
    """
    Validates data integrity before Inference stage.
    Ensures zero-error flow by filtering invalid matches.
    """

    def __init__(self):
        super().__init__("validator")

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate matches and features."""
        matches = context.get("matches", pl.DataFrame())

        if matches.is_empty():
            return {}

        valid_matches = []
        validation_errors = 0

        # Convert to dicts for easier iteration
        match_rows = matches.to_dicts()

        for row in match_rows:
            is_valid, reason = self._validate_match(row)
            if is_valid:
                valid_matches.append(row)
            else:
                validation_errors += 1
                logger.warning(f"Validation failed for {row.get('match_id', '?')}: {reason}")

        if validation_errors > 0:
            logger.info(f"Validator: Dropped {validation_errors} invalid matches.")

        # Reconstruct DataFrame
        if valid_matches:
            new_df = pl.DataFrame(valid_matches)
        else:
            new_df = pl.DataFrame()

        return {"matches": new_df}

    def _validate_match(self, row: Dict[str, Any]) -> tuple[bool, str]:
        """
        Check critical fields.
        """
        match_id = row.get("match_id")
        if not match_id:
            return False, "Missing match_id"

        # 1. Odds Check
        odds = [
            row.get("home_odds"),
            row.get("draw_odds"),
            row.get("away_odds")
        ]

        # Ensure numerical and positive
        for o in odds:
            if not isinstance(o, (int, float)):
                return False, f"Non-numeric odds: {o}"
            if o <= 1.0:
                return False, f"Invalid odds (<=1.0): {o}"

        # 2. Probability Sum Check (Implied Probability)
        # Margin should be reasonable (e.g. 1.0 < sum < 1.2)
        implied_prob = sum(1/o for o in odds)
        if implied_prob < 0.85 or implied_prob > 1.35:
             # Too much vig or arb (arb might be valid but suspicious for basic validation)
             # Let's be lenient for arbs but strict for errors
             if implied_prob < 0.5:
                 return False, f"Implied prob too low ({implied_prob:.2f})"
             if implied_prob > 1.5:
                 return False, f"Implied prob too high ({implied_prob:.2f})"

        # 3. Team Names
        if not row.get("home_team") or not row.get("away_team"):
            return False, "Missing team names"

        return True, "OK"
