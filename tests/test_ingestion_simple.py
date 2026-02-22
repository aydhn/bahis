
import asyncio
import sys
from pathlib import Path
from loguru import logger

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.memory.db_manager import DBManager
from src.ingestion.scraper_agent import ScraperAgent, MackolikScraper, SofascoreScraper
from src.core.data_validator import DataValidator

async def test_ingestion():
    logger.info("--- Starting Ingestion Test ---")
    
    # Setup simple DB (memory or temp)
    db = DBManager(db_path=ROOT / "data" / "test.duckdb")
    
    # 1. Test Validator
    logger.info("Testing DataValidator...")
    validator = DataValidator()
    
    # Case A: Good Data
    good_match = {
        "home_team": "Galatasaray", 
        "away_team": "Fenerbahce", 
        "match_id": "test_1",
        "kickoff": "2024-01-01T20:00:00"
    }
    res = validator.validate_match(good_match)
    if res:
        logger.success(f"Validator accepted valid match: {res['match_id']}")
    else:
        logger.error("Validator REJECTED valid match!")

    # Case B: Partial Data (Soft Validation Check)
    partial_match = {
        "home_team": "Besiktas",
        "away_team": "Trabzonspor"
        # Missing match_id and kickoff
    }
    res = validator.validate_match(partial_match)
    if res and res.get("match_id"):
        logger.success(f"Validator accepted partial match & generated ID: {res['match_id']}")
    else:
        logger.error(f"Validator REJECTED partial match: {validator._errors[-1] if validator._errors else 'Unknown'}")

    # 2. Test Scrapers
    logger.info("Testing Scrapers (Network Request)...")
    
    # Mackolik
    mackolik = MackolikScraper()
    try:
        logger.info("Fetching Mackolik fixtures...")
        fixtures = await mackolik.scrape_fixtures()
        if fixtures:
            logger.success(f"Mackolik returned {len(fixtures)} fixtures. Sample: {fixtures[0]}")
            # Validate one
            v_fixt = validator.validate_match(fixtures[0])
            if v_fixt:
                 logger.success("Mackolik fixture PASSED validation.")
            else:
                 logger.error(f"Mackolik fixture FAILED validation: {validator._errors[-1]}")
        else:
            logger.warning("Mackolik returned 0 fixtures (might be no games or parser fail).")
    except Exception as e:
        logger.error(f"Mackolik crashed: {e}")
    finally:
        await mackolik.close()

    # Sofascore
    sofascore = SofascoreScraper()
    try:
        logger.info("Fetching Sofascore live/scheduled...")
        matches = await sofascore.scrape_live("football")
        if matches:
            logger.success(f"Sofascore returned {len(matches)} matches. Sample: {matches[0]}")
             # Validate one
            v_match = validator.validate_match(matches[0])
            if v_match:
                 logger.success("Sofascore match PASSED validation.")
            else:
                 logger.error(f"Sofascore match FAILED validation: {validator._errors[-1]}")
        else:
            logger.warning("Sofascore returned 0 matches.")
    except Exception as e:
        logger.error(f"Sofascore crashed: {e}")
    finally:
        await sofascore.close()

    logger.info("--- Test Finished ---")

if __name__ == "__main__":
    try:
        asyncio.run(test_ingestion())
    except KeyboardInterrupt:
        pass
