import asyncio
import polars as pl
from typing import Dict, Any
from loguru import logger
from src.pipeline.core import PipelineStage
from src.system.container import container

class IngestionStage(PipelineStage):
    """Fetches and validates upcoming matches for analysis."""

    def __init__(self):
        super().__init__("ingestion")
        self.db = container.get("db")
        # Validator is tricky as it was in `core` but not in container.
        # Let's see if we can instantiate it or if it's a simple class.
        try:
            from src.core.data_validator import DataValidator
            self.validator = DataValidator()
        except ImportError:
            self.validator = None

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Fetch matches from DB."""

        # 1. Get matches
        matches = self.db.get_upcoming_matches()

        if matches.is_empty():
            logger.info("No upcoming matches found. Sleeping 60s.")
            await asyncio.sleep(60)
            return {"matches": pl.DataFrame()}

        # 2. Validate
        if self.validator:
            validated_rows = self.validator.validate_batch(
                matches.to_dicts(), schema="match"
            )
            if not validated_rows:
                logger.warning("All matches failed validation.")
                return {"matches": pl.DataFrame()}
            matches = pl.DataFrame(validated_rows)
            logger.debug(f"Validated {len(validated_rows)} matches.")

        return {"matches": matches}
