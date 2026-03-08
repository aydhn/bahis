import polars as pl
from typing import Dict, Any
from loguru import logger
from src.pipeline.core import PipelineStage
from src.system.container import container
from src.ingestion.mock_generator import MockGenerator
from src.core.circuit_breaker import CircuitBreakerRegistry

class IngestionStage(PipelineStage):
    """Fetches and validates upcoming matches for analysis."""

    def __init__(self):
        super().__init__("ingestion")
        self.db = container.get("db")
        self.mock_gen = MockGenerator()
        self.cb_registry = CircuitBreakerRegistry()
        self.db_breaker = self.cb_registry.get_or_create("db_ingestion", preset="api")

        try:
            from src.core.data_validator import DataValidator
            self.validator = DataValidator()
        except ImportError:
            self.validator = None

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Fetch matches from DB or Fallback to Mock."""

        # 1. Get matches with Circuit Breaker
        matches = pl.DataFrame()

        # Define wrapper for breaker
        def _fetch():
            return self.db.get_upcoming_matches()

        if self.db_breaker.is_available:
            matches = self.db_breaker.call(_fetch)

        if matches is None: # Breaker tripped or call failed
            matches = pl.DataFrame()

        # 2. Mock Fallback (Autonomous Mode)
        mock_features = None
        if matches.is_empty():
            if not self.db_breaker.is_available:
                logger.warning("Circuit Breaker OPEN. Engaging Autonomous Mock Generator.")
            else:
                logger.warning("No upcoming matches in DB. Engaging Autonomous Mock Generator.")
            matches = self.mock_gen.generate_matches(n=10)

            # Generate features immediately for these mock matches
            # This allows InferenceStage to work even if FeatureStage fails to fetch from DB
            if not matches.is_empty():
                match_ids = matches["match_id"].to_list()
                mock_features = self.mock_gen.generate_features(match_ids)
                logger.info(f"Generated {len(matches)} mock matches and features.")

        # 3. Validate (Only validate real matches, Mock is trusted)
        # But if mock data is generated, validator might fail if schema mismatch?
        # Let's skip validation for mock to be safe, or validate if schema matches.
        # Assuming MockGenerator produces valid schema.

        if self.validator and not matches.is_empty():
            # Check if these are mock matches?
            # Mock matches have 'status'="Not Started".
            validated_rows = self.validator.validate_batch(
                matches.to_dicts(), schema="match"
            )
            if not validated_rows:
                logger.warning("All matches failed validation.")
                return {"matches": pl.DataFrame()}
            matches = pl.DataFrame(validated_rows)
            logger.debug(f"Validated {len(validated_rows)} matches.")

        result = {"matches": matches}
        if mock_features is not None:
            result["mock_features"] = mock_features

        return result
