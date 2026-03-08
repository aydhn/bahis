import pytest
import asyncio
from src.core.event_bus import EventBus, Event
from src.quant.risk.portfolio_manager import PortfolioManager

@pytest.mark.asyncio
async def test_flow():
    print("Initializing EventBus...")
    bus = EventBus(persist=False) # Don't write to disk for test
    print("Initializing PortfolioManager...")
    pm = PortfolioManager(bus)

    # Mock data
    pred = {
        "match_id": "test_match_1",
        "selection": "H",
        "odds": 2.0,
        "prob_win": 0.6,
        "confidence": 0.8
    }

    print("Emitting prediction_ready...")
    await bus.emit(Event(event_type="prediction_ready", data=pred))

    await asyncio.sleep(0.1) # Let async handler run

    if len(pm.current_opportunities) == 1:
        print("SUCCESS: Prediction received correctly.")
    else:
        print(f"FAILURE: Expected 1 opportunity, got {len(pm.current_opportunities)}")
        return

    # Test Optimization trigger
    print("Emitting pipeline_cycle_end...")
    # Mock bus.emit again to capture bet_placed
    received_bets = []
    async def capture_bet(event):
        received_bets.append(event.data)

    bus.subscribe("bet_placed", capture_bet)

    await bus.emit(Event(event_type="pipeline_cycle_end"))

    await asyncio.sleep(0.5) # Optimization might take a bit

    if received_bets:
        print(f"SUCCESS: Bets placed: {received_bets}")
        if received_bets[0]["match_id"] == "test_match_1":
            print("SUCCESS: Match ID matches.")
    else:
        print("FAILURE: No bets placed.")

if __name__ == "__main__":
    asyncio.run(test_flow())
