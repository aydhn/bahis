import asyncio
import json
import time
import os
from pathlib import Path

# Mocking settings
class MockSettings:
    DATA_DIR = Path("./test_data")

settings = MockSettings()

# --- Current implementation logic ---
async def current_impl(bet):
    try:
        log_file = settings.DATA_DIR / "paper_trades_sync.jsonl"
        def write_log():
            log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(log_file, "a") as f:
                f.write(json.dumps(bet) + "\n")
        await asyncio.to_thread(write_log)
    except Exception as e:
        print(f"Failed to save paper trade (sync): {e}")

# --- Optimized implementation logic ---
# Note: We simulate the optimized logic, but use a thread pool fallback
# since aiofiles is not installed in this environment.
# In the real code, we use `import aiofiles`.
async def optimized_impl_simulation(bet):
    try:
        log_file = settings.DATA_DIR / "paper_trades_opt.jsonl"
        # In actual code:
        # async with aiofiles.open(log_file, mode="a") as f:
        #     await f.write(json.dumps(bet) + "\n")

        # Simulated logic without the mkdir call on every iteration
        def write_log():
            with open(log_file, "a") as f:
                f.write(json.dumps(bet) + "\n")
        await asyncio.to_thread(write_log)
    except Exception as e:
        print(f"Failed to save paper trade (opt): {e}")

async def run_benchmark(iterations=1000):
    bet = {"match_id": "test_match", "selection": "1", "stake": 100.0, "odds": 2.0}

    # Cleanup and Init
    if settings.DATA_DIR.exists():
        import shutil
        shutil.rmtree(settings.DATA_DIR)
    settings.DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Benchmarking {iterations} iterations...\n")

    # Benchmark Current
    start_time = time.perf_counter()
    for _ in range(iterations):
        await current_impl(bet)
    end_time = time.perf_counter()
    sync_time = end_time - start_time
    print(f"--- Current Implementation (with mkdir) ---")
    print(f"Total time: {sync_time:.4f} seconds")
    print(f"Average time per iteration: {sync_time/iterations:.6f} seconds\n")

    # Benchmark Optimized Simulation
    start_time = time.perf_counter()
    for _ in range(iterations):
        await optimized_impl_simulation(bet)
    end_time = time.perf_counter()
    opt_time = end_time - start_time
    print(f"--- Optimized Implementation Simulation (without mkdir) ---")
    print(f"Total time: {opt_time:.4f} seconds")
    print(f"Average time per iteration: {opt_time/iterations:.6f} seconds\n")

    improvement = (sync_time - opt_time) / sync_time * 100
    print(f"Improvement: {improvement:.2f}%")

if __name__ == "__main__":
    asyncio.run(run_benchmark())
