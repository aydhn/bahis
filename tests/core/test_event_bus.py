import pytest
import asyncio
import time
import os
from unittest.mock import patch, MagicMock

from src.core.event_bus import (
    Event,
    EventFilter,
    EventStore,
    EventBus,
    ReplayEngine,
    match_event,
)

@pytest.fixture
def event_store():
    # Use in-memory SQLite database for tests
    store = EventStore(db_path=":memory:")
    yield store
    store.close()

@pytest.fixture
def event_bus(event_store):
    bus = EventBus(store=event_store, persist=True)
    yield bus

def test_event_creation():
    event = Event(event_type="test", source="scraper", match_id="m1")
    assert event.event_id is not None
    assert event.timestamp > 0
    assert event.event_type == "test"
    assert event.source == "scraper"
    assert event.match_id == "m1"

    data = event.to_dict()
    assert data["event_id"] == event.event_id

def test_match_event_helper():
    event = match_event("goal", "m2", source="model", minute=10)
    assert event.event_type == "goal"
    assert event.match_id == "m2"
    assert event.source == "model"
    assert event.data["minute"] == 10

def test_event_store_append_and_query(event_store):
    e1 = Event(event_type="goal", match_id="m1", timestamp=100)
    e2 = Event(event_type="card", match_id="m1", timestamp=150)
    e3 = Event(event_type="goal", match_id="m2", timestamp=200)

    event_store.append(e1)
    event_store.append(e2)
    event_store.append(e3)

    assert event_store.count() == 3
    assert event_store.count("goal") == 2

    events = event_store.query(EventFilter(event_type="goal"))
    assert len(events) == 2
    assert events[0].match_id == "m1"

    timeline = event_store.get_match_timeline("m1")
    assert len(timeline) == 2
    assert timeline[0].event_type == "goal"
    assert timeline[1].event_type == "card"

def test_event_store_append_batch(event_store):
    events = [
        Event(event_type="e1", timestamp=100),
        Event(event_type="e2", timestamp=200),
    ]
    event_store.append_batch(events)
    assert event_store.count() == 2

@pytest.mark.asyncio
async def test_event_bus_subscribe_emit(event_bus):
    received = []

    def sync_handler(e):
        received.append(e.event_id)

    async def async_handler(e):
        await asyncio.sleep(0.01)
        received.append(e.event_id + "_async")

    event_bus.subscribe("goal", sync_handler)
    event_bus.subscribe("goal", async_handler)

    e = Event(event_type="goal")
    await event_bus.emit(e)

    assert len(received) == 2
    assert e.event_id in received
    assert e.event_id + "_async" in received

    # Store check
    assert event_bus.store.count() == 1

@pytest.mark.asyncio
async def test_event_bus_subscribe_all(event_bus):
    received = []

    def global_handler(e):
        received.append(e)

    event_bus.subscribe_all(global_handler)
    await event_bus.emit(Event(event_type="e1"))
    await event_bus.emit(Event(event_type="e2"))

    assert len(received) == 2

@pytest.mark.asyncio
async def test_event_bus_unsubscribe(event_bus):
    received = []

    def handler(e):
        received.append(e)

    event_bus.subscribe("test", handler)
    await event_bus.emit(Event(event_type="test"))
    assert len(received) == 1

    event_bus.unsubscribe("test", handler)
    await event_bus.emit(Event(event_type="test"))
    assert len(received) == 1

def test_event_bus_emit_sync(event_bus):
    received = []

    def handler(e):
        received.append(e)

    event_bus.subscribe("test", handler)
    event_bus.emit_sync(Event(event_type="test"))

    assert len(received) == 1

@pytest.mark.asyncio
async def test_event_bus_pause_resume(event_bus):
    received = []

    def handler(e):
        received.append(e)

    event_bus.subscribe("test", handler)

    event_bus.pause()
    await event_bus.emit(Event(event_type="test"))
    event_bus.emit_sync(Event(event_type="test"))
    assert len(received) == 0

    event_bus.resume()
    await event_bus.emit(Event(event_type="test"))
    assert len(received) == 1

@pytest.mark.asyncio
async def test_replay_engine_basic(event_bus):
    replayer = ReplayEngine(event_bus)

    received = []
    def handler(e):
        received.append(e)
    event_bus.subscribe("test", handler)

    events = [
        Event(event_type="test", timestamp=100),
        Event(event_type="test", timestamp=200),
    ]

    result = await replayer.replay(events, speed=10.0, real_time=False)

    assert result["events_replayed"] == 2
    assert len(received) == 2
    assert received[0].metadata["is_replay"] is True

@pytest.mark.asyncio
async def test_replay_engine_match(event_bus):
    # Setup data
    e1 = Event(event_type="test", match_id="m1", timestamp=100)
    e2 = Event(event_type="test", match_id="m1", timestamp=200)
    e3 = Event(event_type="test", match_id="m2", timestamp=300)
    event_bus.store.append(e1)
    event_bus.store.append(e2)
    event_bus.store.append(e3)

    replayer = ReplayEngine(event_bus)

    with patch("asyncio.sleep", new_callable=MagicMock) as mock_sleep:
        # We need async sleep mock properly
        async def mock_async_sleep(*args, **kwargs):
            pass
        mock_sleep.side_effect = mock_async_sleep

        result = await replayer.replay_match("m1", speed=10.0)
        assert result["events_replayed"] == 2
        mock_sleep.assert_called()

@pytest.mark.asyncio
async def test_replay_engine_time_range(event_bus):
    # Setup data
    e1 = Event(event_type="test", timestamp=100)
    e2 = Event(event_type="test", timestamp=200)
    e3 = Event(event_type="test", timestamp=300)
    event_bus.store.append_batch([e1, e2, e3])

    replayer = ReplayEngine(event_bus)

    with patch("asyncio.sleep", new_callable=MagicMock) as mock_sleep:
        async def mock_async_sleep(*args, **kwargs):
            pass
        mock_sleep.side_effect = mock_async_sleep

        result = await replayer.replay_time_range(150, 250, speed=10.0)
        assert result["events_replayed"] == 1
