import asyncio
from src.system.digital_twin import DigitalTwin

async def main():
    dt = DigitalTwin()
    report = await dt.dream(n_matches=3)
    print("Report:", report)

if __name__ == "__main__":
    asyncio.run(main())
