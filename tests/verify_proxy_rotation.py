import asyncio
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ingestion.proxy_manager import ProxyManager
from src.ingestion.scraper_agent import BaseScraper
from loguru import logger

async def test_rotation():
    logger.info("Starting Proxy Rotation Test...")
    
    # Initialize Manager
    pm = ProxyManager()
    scraper = BaseScraper("TestScraper", pm)
    
    # 1. First Request
    logger.info("--- Request 1 ---")
    await scraper._ensure_client()
    ip1_resp = await scraper._fetch("http://httpbin.org/ip")
    if ip1_resp:
        logger.info(f"IP 1: {ip1_resp.strip()}")
    else:
        logger.error("Failed to fetch IP 1")

    # 2. Force Rotation
    logger.info("--- Rotating ---")
    await scraper.rotate_session()
    
    # 3. Second Request
    logger.info("--- Request 2 ---")
    await scraper._ensure_client()
    ip2_resp = await scraper._fetch("http://httpbin.org/ip")
    if ip2_resp:
        logger.info(f"IP 2: {ip2_resp.strip()}")
    else:
        logger.error("Failed to fetch IP 2")
        
    await scraper.close()

if __name__ == "__main__":
    asyncio.run(test_rotation())
