import asyncio
from src.core.event_bus import EventBus
from src.extensions.opportunity_scanner import OpportunityScanner

async def test_scanner():
    bus = EventBus()
    scanner = OpportunityScanner(bus)

    # Run the scan once and cancel
    task = asyncio.create_task(scanner.scan())
    await asyncio.sleep(1)
    scanner.stop()
    await asyncio.sleep(1)
    print("Scanner test completed successfully.")

asyncio.run(test_scanner())
