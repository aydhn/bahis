import asyncio
import time
import os
import sys

# Add src to python path so we can import src.system.watchdog
sys.path.insert(0, os.path.abspath("."))

from src.system.watchdog import send_alert

async def block_task():
    """A task that shouldn't be blocked"""
    start = time.perf_counter()
    # We will loop for a short while, expecting no delays
    for i in range(5):
        await asyncio.sleep(0.1)
    return time.perf_counter() - start

async def alert_task():
    """A task that sends an alert"""
    try:
        # Mocking or expecting a timeout/delay if there is no token/chat_id.
        # Actually, let's just test if it blocks.
        # To make requests.post block, we'll hit a real endpoint that takes a while to respond,
        # or we just rely on the API call to Telegram to take some time.
        # Wait, if TELEGRAM_TOKEN is not set, it just returns immediately with a warning.
        # Let's set some dummy token and chat ID, but send it to a non-existent host or something
        # that will timeout after 2 seconds to clearly show the blocking behavior.
        pass
    except Exception as e:
        pass

# Since send_alert uses requests.post which is patched in tests, let's see how it behaves natively.
# Wait, let's patch requests.post in the benchmark to simulate a slow network call.
from unittest.mock import patch
import requests

def slow_post(*args, **kwargs):
    time.sleep(1) # simulate 1s network latency
    class MockResponse:
        pass
    return MockResponse()

async def run_sync_benchmark():
    start = time.perf_counter()
    # We will run 3 alert tasks and 1 block task
    with patch('src.system.watchdog.requests.post', side_effect=slow_post):
        # Setting mock env vars so it proceeds to post
        with patch('src.system.watchdog.TELEGRAM_TOKEN', 'dummy'):
            with patch('src.system.watchdog.TELEGRAM_CHAT_ID', 'dummy'):

                async def wrapper():
                    send_alert("Test sync")

                # The wrapper calls the sync send_alert inside the async loop
                t1 = asyncio.create_task(wrapper())
                t2 = asyncio.create_task(wrapper())
                t3 = asyncio.create_task(wrapper())

                block_t = asyncio.create_task(block_task())

                await asyncio.gather(t1, t2, t3, block_t)

    elapsed = time.perf_counter() - start
    print(f"Total time (sync block): {elapsed:.2f}s")

async def run_async_benchmark():
    # After refactor, send_alert will be async.
    # We will try to await it. If it's still sync (before refactor), it will fail.
    # So we'll use a dynamic check.
    start = time.perf_counter()
    with patch('src.system.watchdog.TELEGRAM_TOKEN', 'dummy'):
        with patch('src.system.watchdog.TELEGRAM_CHAT_ID', 'dummy'):
            # If it's async, we mock httpx.AsyncClient.post
            try:
                import httpx
                class AsyncMockResponse:
                    pass
                async def slow_async_post(*args, **kwargs):
                    await asyncio.sleep(1)
                    return AsyncMockResponse()

                with patch('httpx.AsyncClient.post', side_effect=slow_async_post):
                    async def wrapper():
                        from src.system.watchdog import send_alert
                        if asyncio.iscoroutinefunction(send_alert):
                            await send_alert("Test async")
                        else:
                            # It's still sync, let's simulate sync slow_post
                            with patch('src.system.watchdog.requests.post', side_effect=slow_post):
                                send_alert("Test sync")

                    t1 = asyncio.create_task(wrapper())
                    t2 = asyncio.create_task(wrapper())
                    t3 = asyncio.create_task(wrapper())
                    block_t = asyncio.create_task(block_task())

                    await asyncio.gather(t1, t2, t3, block_t)
            except ImportError:
                pass

    elapsed = time.perf_counter() - start
    print(f"Total time (async): {elapsed:.2f}s")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "async":
        asyncio.run(run_async_benchmark())
    else:
        asyncio.run(run_sync_benchmark())
